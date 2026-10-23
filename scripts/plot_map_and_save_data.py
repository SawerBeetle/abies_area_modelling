import os

import json
import matplotlib.pyplot as plt
import numpy as np

from plot_map import plot_map

# загрузить конфиг
with open('C:/Users/user/Yandex.Disk/Важные документы/Исходные данные для статей/Моделирование вспышек/Моделирование ареала пихты/Abies_01/config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# загрузить целевую метрику
TARGET_METRIC = config['TARGET_METRIC']
# загрузить адрес корневой директории
ROOT_DIR = config['ROOT_DIR']
# будет ли обучаться модель
TRAIN_MODEL = config['TRAIN_MODEL']

def plot_map_and_save_data(for_map, predictors, used_predictors, model, pred_test, path_to_folder):
    # Создадим фрейм для картирования из признаков, используемых моделью, координат точек 
    # и данных об отсутствии/присутствии пихты ('vegetation'). 
    for_map = for_map[[*used_predictors['feature'], 'point_x', 'point_y', 'vegetation']].copy()
    # переименуем столбец с данными о растительности по Барталёву в 'vegetation_real'
    for_map.rename(columns={'vegetation': 'vegetation_real'}, inplace=True)
    # создадим столбец с результатами работы модели
    for_map['vegetation_predicted'] = model.predict(
        # `[list(pred_test.columns)]` нужно, чтобы упорядочить столбцы 
        # согласно требованиям модели. 
        predictors[[*used_predictors['feature']]][list(pred_test.columns)]
        )

    
    # создадим пустой столбец для оценок результатов работы модели
    for_map['prognosis_res'] = ''

    # задаем список условий (логические маски для всей таблицы)
    conditions = [
        (for_map['vegetation_real'] == 1) & (for_map['vegetation_predicted'] == 1),  # TP
        (for_map['vegetation_real'] == 1) & (for_map['vegetation_predicted'] == 0),  # FN
        (for_map['vegetation_real'] == 0) & (for_map['vegetation_predicted'] == 0),  # TN
        (for_map['vegetation_real'] == 0) & (for_map['vegetation_predicted'] == 1)   # FP
    ]
    # задаем соответствующие им значения
    choices = ['TP', 'FN', 'TN', 'FP']
    # применяем махом ко всему датафрейму (без циклов!)
    for_map['prognosis_res'] = np.select(conditions, choices, default='')
    
    # создаём фигуру для карт и подграфики
    fig, ax = plt.subplots(ncols=3, figsize=(16, 6))
    # Выведем три карты: реальные данные о растительности, ... 
    plot_map(
        ax[0], 
        for_map.sort_values(by='vegetation_real', ascending=False), 
        'vegetation_real', 
        'real data', 
        ['purple', 'green'], 
        ['fir', 'other']
        )
    # ...прогнозные данные о растительности и...
    plot_map(
        ax[1], 
        for_map.sort_values(by='vegetation_predicted', ascending=False), 
        'vegetation_predicted', 
        'predicted data', 
        ['purple', 'green'], 
        ['fir', 'other']
        )
    # ...оценка работы прогноза (совпадение реальных и прогнозных данных). 
    plot_map(
        ax[2], 
        for_map.sort_values(by='prognosis_res'), 
        'prognosis_res', 
        'confusions', 
        ['yellow', 'magenta', 'green', 'purple']
        );
    """ 
    Сортировка по столбцу с отрисовываемыми данными добавлена по совету Google AI, 
    потому что без неё точки окрашиваются не в те цвета; это зависит от порядка размещения 
    данных в строках, и верный результат получается только при их упорядочивании.  
    """
    plt.tight_layout()

    if TRAIN_MODEL:
        # зададим папку для сохранения
        os.chdir(path_to_folder)
        # сохраним карту
        plt.savefig('map.jpeg')

        # сохраним 'for_map' как .csv
        os.chdir(path_to_folder)
        for_map.to_csv(
            path_or_buf='data_for_map.csv', 
            index=False
            )

    