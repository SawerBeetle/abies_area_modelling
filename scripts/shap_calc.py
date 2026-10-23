from IPython.display import display
import os
import pathlib
from pathlib import Path
import pickle
from typing import Literal

import json
import lightgbm as lgb
import matplotlib.colors as mcolors
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
    # режим работы функции
    mode: Literal['beeswarm', 'scatter'], 
    # первый сопоставляемый признак (при mode == 'scatter')
    feature1: str = None, 
    # второй сопоставляемый признак (при mode == 'scatter')
    feature2: str = None, 
    # названия признаков для создания удобочитаемых подписей
    feature_name1: str = None, 
    feature_name2: str = None,
    # значения вклада признаков в прогноз
    shap_values: shap._explanation.Explanation | np.ndarray | None = None, 
    ):

    # отофрматируем пути к корневому каталогу и каталогу с моделью
    root_dir = Path(ROOT_DIR)
    model_path = Path(str(model_path).lstrip(r'\/'))


    # загрузим модель
    try: 
        with open(root_dir / model_path / model_file, 'rb') as m: 
            model = pickle.load(m)
    except Exception as e: 
        print(f"Тип ошибки: {type(e)}")
        raise  # Пробрасываем ошибку дальше

    display(model)

    # создадим список столбцов в порядке, который примет модель
    columns_in_right_order = [str(feature) for feature in model.feature_names_in_]

    # Создадим наборы данных для пихты и прочих пород
    # с нужными (для модели) столбцами в нужном (для модели) порядке. 
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
    data_for_shap = pd.concat(
        [data_for_shap_abies, data_for_shap_not_abies], 
        axis=0, 
        ignore_index=True
        )

    if shap_values is None: 
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
            # создаём полотно с двумя субграфиками
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1, 2]})

            # активируем первый субграфик
            plt.axes(ax1)
            # выводим в него столбчатый график значений вклада признаков в модель
            shap.summary_plot(
                # значения вклада признаков
                shap_values, 
                # исходные данные (для подписей столбцов)
                features=data_for_shap, 
                # задаём столбчатый тип графика
                plot_type='bar', 
                # блокирует немедленный вывод на экран, чтобы показать его вместе со вторым графиком
                show=False, 
                # указывает, что размеры субграфика будут рассчитаны автоматически
                plot_size=None
                )
            # устанавливаем обозначение субграфика
            ax1.set_title("A", fontsize=20, pad=10)
            # подписываем ось
            ax1.set_xlabel("Mean absolute value of importance (log-odds)", fontsize=18)
            # форматируем цифры на оси абсцисс
            ax1.tick_params(axis='x', labelsize=16)  
            # подписываем названия признаков на оси ординат
            ax1.tick_params(axis='y', labelsize=16) 

            # активируем второй субграфик
            plt.axes(ax2)
            # строим в нём SHAP-график
            shap.plots.beeswarm(shap_values, alpha=0.05, show=False, plot_size=None)
            # устанавливаем обозначение субграфика
            ax2.set_title("B", fontsize=20, pad=10)
            # подписываем ось
            ax2.set_xlabel("SHAP value (impact on model output)", fontsize=18)
            # форматируем цифры на оси абсцисс
            ax2.tick_params(axis='x', labelsize=18) 
            # подписываем названия признаков на оси ординат
            ax2.tick_params(axis='y', labelsize=18) 

            # настройка шрифта для цветовой шкалы (colorbar) на графике B
            # SHAP создает шкалу как отдельную ось в рамках текущей фигуры
            try:
                # создаём из цветовой шкалы (fig.axes[-1]) отдельный объект и форматируем его
                colorbar_ax = fig.axes[-1]
                # цифры на цветовой шкале (Low / High)
                colorbar_ax.tick_params(labelsize=18) 
                # текст шкалы
                colorbar_ax.set_ylabel("Feature value", fontsize=17) 
            except Exception:
                pass

            # оптимизируем положение объектов на графике, чтобы не было наложений
            plt.tight_layout()
            # зададим путь для сохранения (в папку с моделью)
            savepath = root_dir / model_path / 'shap.jpeg'
            # сохраним график
            plt.savefig(savepath, dpi=600)
            # выведем график на экран
            plt.show()

            # вернём значения вклада признаков в прогноз, чтобы не считать по-новой
            return shap_values
        
        # если 'mode='scatter'', строим график рассеяния SHAP
        else: 
            # задаем три диапазона значений
            num_colors = 3
            # задаём цветовую шкалу (из проверенных plasma оказалась наиболее наглядной)
            base_cmap = plt.get_cmap("plasma")
            # Извлекаем список дискретных цветов в виде массива numpy, 
            # содержащего значения RGB этих цветов. 
            color_list = base_cmap(np.linspace(0, 1, num_colors))
            # создаём дискретную цветовую шкалу, основанную на непрерывной 'plasma'
            discrete_cmap = mcolors.ListedColormap(color_list)

            # создаём полотно графика
            plt.figure(figsize=(8, 5))
            # создаем график рассеяния SHAP
            shap.plots.scatter(
                # значения точек
                shap_values[:, feature1], 
                # цвет точек (определяется значениями второго признака, с которым мы сопоставляем первый)
                color=shap_values[:, feature2], 
                # Прозрачность; 0.5 оптимально, чтобы был виден преобладающий цвет точек, 
                # но при этом он не забивал бы более редкие цвета. 
                alpha=0.5, 
                # цветовая палитна
                cmap=discrete_cmap, 
                # блокирует немедленный вывод на экран, чтобы показать его вместе со вторым графиком
                show=False
            )
            # тонкая настройка графики через matplotlib
            plt.title(f"Dependence of {feature_name2} from {feature_name1}", fontsize=14, pad=15)
            # подписываем ось абсцисс
            plt.xlabel(f"{feature_name1} (real values)", fontsize=12)
            # подписываем ось ординат
            plt.ylabel("SHAP Log-Odds", fontsize=12)
            # создаём сетку
            plt.grid(True, linestyle="--", alpha=0.5)
            # оптимизируем положение объектов на графике, чтобы не было наложений
            plt.tight_layout()
            # выводим график на экран
            plt.show()

            # вернём значения вклада признаков в прогноз, чтобы не считать по-новой
            return shap_values
        
    except Exception as e: 
        print(f'Ошибка {e}; проверьте правильность ввода режима работы функции (mode). ')

    
