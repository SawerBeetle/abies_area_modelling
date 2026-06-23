import os

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram

# загрузить конфиг
with open('C:/Users/user/Yandex.Disk/Важные документы/Исходные данные для статей/Моделирование вспышек/Моделирование ареала пихты/Abies_01/config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# Загрузить максимально допустимое значение VIF, корневую папку и целевую метрику, 
# обозначить, будет ли сохранена дендрограмма в папку 'eda' (EDA). 
EDA = config['EDA']
MAX_VIF = config['MAX_VIF']
ROOT_DIR = config['ROOT_DIR']
TARGET_METRIC = config['TARGET_METRIC']
TRAIN_MODEL = config['TRAIN_MODEL']

def remove_multicollinear(
        # полный набор данных для исследования
        data, 
        # обучающий, валидационный и тестовый наборы для удаления из них избыточных предикторов
        train, 
        valid, 
        test, 
        # метка даты и времени для создания пути к лог-файлу
        dt, 
        # имя лог-файла
        fname, 
        # максимально допустимое значение VIF; признаки, у которых оно выше, удаляются
        allowed_vif=MAX_VIF
        ): 
    # подготовим список с метками классов
    vegetation = [
        *np.ones(int(data.shape[0] / 2)).tolist(), 
        *np.zeros(int(data.shape[0] / 2)).tolist()
        ]
    
    # Создадим фрейм для кластеризации, объединив данные для пихты и 
    # прочих сообществ. Также удалим из него данные о координатах и ID, 
    # сбросим NaN. 
    for_clustering = data.drop(['point_x', 'point_y'], axis=1).T.dropna(axis=1)

    # выполним кластеризацию стандартизированных данных по методу полного связывания
    clustering_res = linkage(for_clustering, metric='correlation', method='complete')

    if TRAIN_MODEL:
        # отобразим результаты кластеризации графически
        # выведем поясняющее сообщение
        print('Группировка предикторов: ')
        # создадим фигуру (контейнер для дендрограммы)
        plt.figure(figsize = (12, 4.5))
        # создадим собстенно дендрограмму
        dendrogram_weather = dendrogram(
            # результаты кластеризации
            clustering_res, 
            # метки классифицируемых признаков на оси абсцисс
            labels=for_clustering.index, 
            # граница разбиения (и окрашивания) кластеров
            color_threshold=0.7
            )

        plt.tight_layout()

        # сохраняем дендрограмму
        if EDA: 
            if os.path.exists(os.path.join(ROOT_DIR + '/eda')): 
                os.chdir(ROOT_DIR + '/eda')
                plt.savefig('feature_clusters.jpeg')
            else: 
                print('Папка для сохранения дендрограммы не существует.')

        # вывод дендрограммы на экран
        plt.show()

    # готовим матрицу признаков для процедуры удаления мультиколлинеарности
    df_vif = for_clustering.T

    while True:
        # считаем все VIF разом через инверсию матрицы
        corr = df_vif.corr()
        vif_values = np.diag(np.linalg.inv(corr.values))
        
        max_vif = vif_values.max()
        max_idx = vif_values.argmax()
        
        if max_vif <= allowed_vif:
            break
            
        excluded_var = df_vif.columns[max_idx]
        print(f'\rУдаляем: {excluded_var} (VIF={max_vif:.2f})', end='')
        
        # удаляем только один худший признак
        df_vif = df_vif.drop(columns=[excluded_var])

    # итоговый результат (оставшиеся переменные и соответствующие значения VIF)
    vif_data = pd.DataFrame(
        {
            "Variable": df_vif.columns, 
            "VIF": np.diag(np.linalg.inv(df_vif.corr().values))
            }
        ).sort_values(by='VIF', ascending=False)

    # оставим в выборках только те столбцы, которые не дают избытка VIF
    pred_train = train[list(vif_data['Variable'])]
    pred_test = test[list(vif_data['Variable'])]
    pred_valid = valid[list(vif_data['Variable'])]

    # создадим данные для разведочного анализа
    for_ydp = data.copy()
    for_ydp = for_ydp[list(vif_data['Variable'])]
    for_ydp['veg'] = vegetation

    # запишем данные в лог-файл
    try:
        # зададим рабочую папку
        os.chdir(ROOT_DIR + '/models/abies_area_model_' + dt + '_' + TARGET_METRIC)
        with open(fname, 'a') as file: 
            file.write('\n')
            # запишем максимально допустимое значение VIF
            file.write(f'Максимальное значение VIF: {MAX_VIF}. \n')
            # запишем переменные, оставшиеся после удаления мультиколлинеарных данных
            file.write('Переменные, оставшиеся после удаления мультиколлинеарных данных: \n')
            file.write('    ' + ', '.join(list(pred_train.columns)) + '. \n')
    except:
        print()
        print('Проблема с логированем списка переменных.')

    return pred_train, pred_valid, pred_test, for_ydp, vif_data