import os
import pickle
import string

import json
import matplotlib.pyplot as plt
import math
import pandas as pd
import seaborn as sns

# загрузить конфиг
with open('C:/Users/user/Yandex.Disk/Важные документы/Исходные данные для статей/Моделирование вспышек/Моделирование ареала пихты/Abies_01/config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# загрузить путь к корневой папке
ROOT_DIR = config['ROOT_DIR']

def plot_future_climate_partially(
        # словарь с данными о климате
        cmip_dict: dict, 
        # путь к бинарному файлу с результатами прогноза
        predictions_path: str, 
        # путь к файлу для сохранения графика
        target_path: str, 
        # заголовок графика
        title: str, 
        # количество столбцов для субграфиков
        n_cols=3
        ):

    # извлечём ключи (характеристики погоды) из обрабатываемого слованя
    keys0 = list(cmip_dict.keys())

    # загрузим результаты прогнозирования для данной траектории SSP
    try: 
        with open(os.path.join(ROOT_DIR, predictions_path), 'rb') as f: 
            # загрузка
            # NB: подразумевается, что загружаемый файл должен являться таблицей, где 
            # каждая строка – это полигон (пиксель) в пределах исследуемой территории, 
            # а каждый столбец – результат n-й симуляции, которая даёт для каждого пикселя 
            # значение 0 (условий для пихты нет) или 1 (условия для пихты подходящие). 
            predictions = pickle.load(f)
            # расчёт вероятности
            predictions = predictions.mean(axis=1) 
    except FileNotFoundError as e: 
        raise FileNotFoundError(
        print('Файл с результатами симуляций отсутствует; выполните код для его создания или вызовите эту функцию после ячейки, в которой он выполняется.')
        ) from e
            
    # зададим ширину графика
    width = 10.5
    # рассчитаем нужное количество строк и высоту графика
    n_rows = math.ceil(len(cmip_dict) / n_cols)
    height = 3 * n_rows

    # зададим базовый размер шрифта
    font_size = 12

    # создадим полотно для графиков
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(width, height))
    # сделаем нумерацию субграфиков одномерной
    ax_flat = ax.flatten()

    # определим счётчик
    k = 0
    # получим список заглавных букв для заголовков субграфиков
    en_letters = list(string.ascii_uppercase)

    # В цикле создадим и добавим на полотно субграфики, 
    # последовательно обработав сценарии SSP из словаря 'cmip_dict'. 
    for key0, axis in zip(keys0, ax_flat): 
        # извлечём из словаря для обрабатываемого сценария ключи, содержащие имена моделей
        keys1 = list(cmip_dict[key0])
        # Список для хранения средних значений по всем моделям на текущем субграфике; 
        # '_0' - для точек с нулевой вероятностью доминирования пихты, '_1' - с ненулевой. 
        all_models_data_0 = []
        all_models_data_1 = []     

        # последовательно обработаем модели, данные для которых находятся в 'cmip_dict[key0]'
        for key1 in keys1: 

            # Вычисляем средние по моделям для точек 
            # с нулевой вероятностью доминирования пихтовых древостоев '_0' и с ненулевой '_1'. 
            model_mean_0 = cmip_dict[key0][key1].loc[predictions == 0, :].mean(axis=0)
            model_mean_1 = cmip_dict[key0][key1].loc[predictions > 0, :].mean(axis=0)
            # сохраняем данные модели для последующего общего усреднения
            all_models_data_0.append(model_mean_0)
            all_models_data_1.append(model_mean_1)

        # --- ДОБАВЛЕНИЕ УСРЕДНЯЮЩЕЙ ЛИНИИ ДЛЯ ВСЕХ МОДЕЛЕЙ ---
        # для точек с нулевой вероятностью доминирования пихты
        if all_models_data_0:
            # превращаем список данных во многомерный массив и находим среднее по строкам (axis=0)
            total_mean_0 = pd.DataFrame(all_models_data_0).mean(axis=0)
            # сглаживаем полученную линию скользящей средней
            total_mean_0 = total_mean_0.rolling(window=5, min_periods=1, center=True).mean()

            # строим усредняющую линию
            sns.lineplot(
                data=total_mean_0,
                ax=axis,
                lw=3.0, 
                color='green', 
                # подпись в легенде
                label='Mean (zero probability)', 
                # чтобы легенда не выводилась для каждого субграфика
                legend=False
            )
        # для точек с ненулевой вероятностью доминирования пихты
        if all_models_data_1:
            # превращаем список данных во многомерный массив и находим среднее по строкам (axis=0)
            total_mean_1 = pd.DataFrame(all_models_data_1).mean(axis=0)
            # сглаживаем полученную линию скользящей средней
            total_mean_1 = total_mean_1.rolling(window=5, min_periods=1, center=True).mean()

            # строим усредняющую линию
            sns.lineplot(
                data=total_mean_1,
                ax=axis,
                lw=3.0, 
                color='purple', 
                # подпись в легенде
                label='Mean (non-zero probability)', 
                # чтобы легенда не выводилась для каждого субграфика
                legend=False
            )
        # -----------------------------------------------------

        # добавляем буквы
        axis.set_title(
            label=en_letters[k], 
            # прижимает текст к левому краю
            loc='left',          
            fontweight='bold', 
            fontsize=font_size + 4
        )
        # задаём подпись оси абсцисс
        axis.set_xlabel('Year', fontsize=font_size + 2)
        # задаём подпись оси ординат (характеристика погоды)
        axis.set_ylabel(key0, fontsize=font_size + 2)
        # устанавливаем характеристики засечек при осях
        axis.tick_params(axis='both', labelsize=font_size)
        # обновим счётчик
        k += 1

    # устанавливаем общий заголовок
    fig.suptitle(title, fontsize=font_size + 6, y=0.98)

    # --- СОЗДАНИЕ ЕДИНОЙ ОБЩЕЙ ЛЕГЕНДЫ ДЛЯ ВСЕЙ ФИГУРЫ ---
    # берём линии и подписи с первого субграфика (ax_flat[0])
    handles, labels = ax_flat[0].get_legend_handles_labels()

    # добавляем общую легенду на саму фигуру (fig) справа от графиков
    fig.legend(
        # графические объекты (элементы легенды)
        handles, 
        # подписи к ним
        labels, 
        loc='upper center', 
        # координаты: чуть правее правого края полотна
        bbox_to_anchor=(0.5, 0.05), 
        fontsize=font_size + 8, 
        frameon=True
    )

    # предотвратим наложение субграфиков
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    # сохраним графики в папку с моделью
    plt.savefig(
        os.path.join(ROOT_DIR, target_path), 
        dpi=300, 
        bbox_inches='tight'
        )
    # выведем графики на экран
    plt.show()


