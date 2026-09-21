from IPython.display import display
import os
import pathlib
import pickle
from typing import Literal

import json
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import shap
import xgboost as xgb

# загрузить конфиг
with open('C:/Users/user/Yandex.Disk/Важные документы/Исходные данные для статей/Моделирование вспышек/Моделирование ареала пихты/Abies_01/config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# адрес корневой папки
ROOT_DIR = config['ROOT_DIR']
# зерно для генератора случайных чисел
SEED = config['SEED']

def shap_calc(
    # набор данных для полигонов с доминированием пихты
    data_abies: pd.DataFrame, 
    # набор данных для полигонов с доминированием других поролд
    data_other: pd.DataFrame, 
    # имя файла модели
    model_file: str, 
    # путь к файлу модели
    model_path: str | pathlib.Path,
    # значения вклада признаков в прогноз
    shap_values: shap._explanation.Explanation | None, 
    # режим работы функции
    mode: Literal['beeswarm', 'scatter'], 
    # первый сопоставляемый признак (при mode == 'scatter')
    feature1: str = None, 
    # второй сопоставляемый признак (при mode == 'scatter')
    feature2: str = None, 
    # названия признаков для создания удобочитаемых подписей
    feature_name1: str = None, 
    feature_name2: str = None
    ):

    # загрузим модель
    os.chdir(ROOT_DIR + '/' + model_path)
    model = pickle.load(open(model_file, 'rb'))
    display(model)

    # создадим список столбцов в порядке, который примет модель
    columns_in_right_order = [str(feature) for feature in model.feature_names_in_]

    # Создадим наборы данных для пихты и прочих пород
    #  с нужными (для модели) столбцами в нужном (для модели) порядке. 
    data_for_shap_abies = data_abies.loc[:, columns_in_right_order]
    data_for_shap_not_abies = data_other.loc[:, columns_in_right_order]

    # Создадим генератор случайных чисел для выбора строк (полигонов)
    # из набора данных для прочих пород. 
    # Это нужно, т.к. объектов слишком уж много, график будет строиться часами. 
    rng = np.random.default_rng(seed=SEED)
    # зададим количество выбираемых объектов
    num_rows_to_select = data_for_shap_not_abies.shape[0] // 10

    # выберем полигоны с доминированием прочих пород
    data_for_shap_not_abies = rng.choice(
        # объект, из которого делаем выборку
        data_for_shap_not_abies, 
        # объём данных
        size=num_rows_to_select, 
        # выборка рядов
        axis=0, 
        # выборка без возвращения
        replace=False
        )
    # Конвертируем массив данных о полигонах
    # с доминированием прочих пород в Pandas data frame. 
    data_for_shap_not_abies = pd.DataFrame(
        data_for_shap_not_abies, 
        columns=list(data_for_shap_abies.columns)
        )

    # Объединим наборы данных для полигонов с доминированием пихты и прочих пород
    # в единый фрейм для дальнейшего создания графиков на его основе. 
    data_for_shap = pd.concat([data_for_shap_abies, data_for_shap_not_abies], axis=0)

    if not shap_values: 
        # создадим объект ('explainer') для расчёта важности признаков
        # HACK: Используем KernelExplainer вместо TreeExplainer, т.к. при текущих версиях
        # библиотек более продвинутый и быстрый TreeExplainer работать не будет. 
        explainer = shap.KernelExplainer(
            # HACK: Лямбда-функция нужна, т.к. при текущих версях библиотек
            # без неё explainer работать не будет. 
            lambda x: model.predict(x), shap.sample(data_for_shap, 100)
        )
        # получим значения вклада признаков в прогноз
        shap_values = explainer(data_for_shap)

    try: 
        # для графика вклада признаков в прогноз
        if mode == 'beeswarm': 
            # построим график значимости признаков
            plt.figure(figsize=(10, 6))
            # собственно построение
            shap.plots.beeswarm(shap_values, alpha=0.05, show=False)
            # оптимизируем положение объектов на графике, чтобы не было наложений
            plt.tight_layout()
            # зададим путь для сохранения (в папку с моделью)
            savepath = pathlib.Path(ROOT_DIR) / model_path / 'shap.jpeg'
            # сохраним график
            plt.savefig(savepath, dpi=600)
            # выведем график на экран
            plt.show()
            # вернём значения вклада признаков в прогноз, чтобы не считать по-новой
            return shap_values
        else: 
            # зависимость признаков друг от друга
            plt.figure(figsize=(8, 5))
            # создаем график рассеяния SHAP
            shap.plots.scatter(
                shap_values[:, feature1], color=shap_values[:, feature2], alpha=0.5, show=False
            )
            # Тонкая настройка графики через matplotlib
            plt.title(f"Dependence of {feature_name2} from {feature_name1}", fontsize=14, pad=15)
            plt.xlabel(f"{feature_name1} (real values)", fontsize=12)
            plt.ylabel("SHAP Log-Odds", fontsize=12)
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.show()
            # вернём значения вклада признаков в прогноз, чтобы не считать по-новой
            return shap_values
    except Exception as e: 
        print(f'Ошибка {e}; проверьте правильность ввода режима работы функции (mode). ')

    
