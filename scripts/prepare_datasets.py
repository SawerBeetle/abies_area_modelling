import os

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# загрузить конфиг
with open('C:/Users/user/Yandex.Disk/Важные документы/Исходные данные для статей/Моделирование вспышек/Моделирование ареала пихты/Abies_01/config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# загрузить адрес корневой папки
ROOT_DIR = config['ROOT_DIR']
# загрузить SEED
SEED = config['SEED']
# загрузить целевую метрику
TARGET_METRIC = config['TARGET_METRIC']

def prepare_datasets(
        # фрейм с данными по пихтовым древостоям
        data_abies, 
        # фрейм с данными по древостоям других пород
        data_not_abies, 
        # метка даты и времени (нужна для создания пути к лог-файлу)
        dt, 
        # имя лог-файла
        fname
        ): 
    # объединим фреймы с предикторами (пихта сверху)
    predictors = pd.concat([data_abies, data_not_abies], axis=0)

    # подготовим список с метками классов
    vegetation = [
        *np.ones(int(predictors.shape[0] / 2)).tolist(), 
        *np.zeros(int(predictors.shape[0] / 2)).tolist()
        ]
    
    # Скопируем данные о предикторах во фрейм 'for_map', 
    # который далее понадобится для составления карт растительности.
    for_map = predictors.copy()
    # добавим в него данные о растительности (1 - пихта, 0 - прочее)
    for_map['vegetation'] = vegetation
    # т.к. индексы для строк с 1 и 0 могут дублироваться, заменим их на уникальные
    for_map.index = list(range(0, for_map.shape[0]))

    """ 
    Разделим данные на обучающие (train) и промежуточные (interim) 
    для дальнейшего деления на валидационную и тестовую выборки. 
    Соотношение train:interim установим 0.5:0.5. 
    """
    pred_train, pred_interim, veg_train, veg_interim = train_test_split(
        predictors, vegetation, train_size=0.5, random_state=SEED
        ) 

    # Проверим результат
    print(
        'Размер обучающей выборки ', pred_train.shape[0], 
        ' объектов, размер промежуточной – ', pred_interim.shape[0], 
        ' объектов.', sep=''
        )
    
    # Теперь разделим промежуточную выборку (interim) на 
    # валидационную (valid) и тестовую (test) в соотношении 0.5:0.5. 
    pred_valid, pred_test, veg_valid, veg_test = train_test_split(
        pred_interim, veg_interim, train_size=0.5, random_state=SEED
        )

    # Проверим результат
    print(
        'Размер валидационной выборки ', pred_valid.shape[0], 
        ' объектов, размер тестовой – ', pred_test.shape[0], 
        ' объектов.', sep=''
        )
    
    print('Доля участков с преобладанием пихты в обучающем наборе: {:5.4G}.'.format(np.mean(veg_train)), sep='')
    print('Доля участков с преобладанием пихты в валидационном наборе: {:5.4G}.'.format(np.mean(veg_valid)), sep='')
    print('Доля участков с преобладанием пихты в тестовом наборе: {:5.4G}.'.format(np.mean(veg_test)), sep='')
    
    # запишем некоторые данные в лог-файл
    try: 
        os.chdir(ROOT_DIR + '/models/abies_area_model_' + dt + '_' + TARGET_METRIC)
        with open(fname, 'a') as file: 
            file.write(f'Объём обучающей выборки составил {pred_train.shape[0]} объектов. \n')
            file.write(f'Объём валидационной выборки составил {pred_valid.shape[0]} объектов. \n')
            file.write(f'Объём тестовой выборки составил {pred_test.shape[0]} объектов. \n')
    except: 
        print('Проблемы с логированием объёмов выборок')    

    return for_map, pred_train, pred_valid, pred_test, veg_train, veg_valid, veg_test