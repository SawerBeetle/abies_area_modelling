import os
import re
import winsound

import json
import math
import numpy as np
import pandas as pd
from tqdm import auto

from weather_means import weather_means

# загрузить конфиг
with open('C:/Users/user/Yandex.Disk/Важные документы/Исходные данные для статей/Моделирование вспышек/Моделирование ареала пихты/Abies_01/config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# загрузить список географических признаков
geogr_feat = config['GEOGRAPHICAL_FEATURES']
# загрузить список номеров месяцев
month_num = config['MONTH_NUMBERS']
# загрузить названия файлов с данными по пихте и прочим лесообразователям
NAME_ABIES_FILE = config['NAME_ABIES_FILE']
NAME_OTHERSP_FILE = config['NAME_OTHERSP_FILE']
# загрузить адрес корневой папки
ROOT_DIR = config['ROOT_DIR']
# загрузить список характеристик погоды
weather_feat = config['WEATHER_FEATURES']

def load_weather_completely(
        abies_file=NAME_ABIES_FILE, 
        other_file=NAME_OTHERSP_FILE, 
        root_dir=ROOT_DIR, 
        chunk_volume=10000, 
        model=None, 
        mode='abies'
        ): 
    # если обрабатываем данные для пихтовых древостоев
    if mode == 'abies': 
        # зададим путь к файлу с данными
        path_to_abies = os.path.join(root_dir, 'data', abies_file)
        # считаем количество строк для пихты
        with open(path_to_abies, 'r') as f:
            # вычитаем 1, т.к. не нужно учитывать строку заголовка
            row_count_abies = sum(1 for line in f) - 1 
            print(f"Количество строк с данными для пихты: {row_count_abies}")
            # рассчитываем количество чанков
            abies_chunk_num = math.ceil(row_count_abies / chunk_volume)
            print(f'Количество чанков при {chunk_volume} строк на чанк: {abies_chunk_num}.')
            print()
    # если обрабатываем данные для других пород
    elif mode == 'other':
        # зададим путь к файлу с данными
        path_to_not_abies = os.path.join(root_dir, 'data', other_file)
        # считаем количество строк для других пород
        with open(path_to_not_abies, 'r') as f:
            # вычитаем 1, т.к. не нужно учитывать строку заголовка
            row_count_not_abies = sum(1 for line in f) - 1 
            print(f"Количество строк с данными для других лесообразователей: {row_count_not_abies}")
            # рассчитываем количество чанков
            not_abies_chunk_num = math.ceil(row_count_not_abies / chunk_volume)
            print(f'Количество чанков при {chunk_volume} строк на чанк: {not_abies_chunk_num}.')
    # при ошибочном выборе режима работы выводим сообщение об этом
    else: 
        print('Выберите значение mode, равное abies или other.')

    # проверка наличия целевой папки и создание, если её нет
    if not os.path.exists(os.path.join(root_dir, 'map_current')): 
        os.mkdir(os.path.join(root_dir, 'map_current'))

    # создаём ридер для пихты и прогресс-бар
    if mode == 'abies': 
        data_reader = pd.read_csv(path_to_abies, sep=';', decimal=',', chunksize=chunk_volume)
        pbar = auto.tqdm(data_reader, total=abies_chunk_num, leave=False)
    # создаём ридер для прочих лесообразователей и прогресс-бар
    else: 
        data_reader = pd.read_csv(path_to_not_abies, sep=';', decimal=',', chunksize=chunk_volume)
        pbar = auto.tqdm(data_reader, total=not_abies_chunk_num, leave=False)

    # создаём пустой список для результатов работы модели
    predictions = []

    # загружаем и обрабатываем данные чанками размера 'chunk_volume'
    for data in pbar: 
        # добавляем к номеру текущей итерации единицу для удобочитаемости
        current_chunk = pbar.n + 1
        # апдейтим описание прогресс-бара
        if mode == 'abies': 
            pbar.set_description_str(f'Чанк {current_chunk} из {abies_chunk_num}')
        else: 
            pbar.set_description_str(f'Чанк {current_chunk} из {not_abies_chunk_num}')
        # сохраним названия столбцов
        colnames = data.columns

        # Приведём названия столбцов во фреймах с исходными данными к единообразному виду. 
        # Во-первых, заменим цифровые коды (b1...b1028) на годы и месяцы. Во-вторых, 
        # приведём названия столбцов к snake_case. Естественно, всё это выполняется только 
        # Для начала создадим список дат в формате "год_месяц" начиная с января 1940 года. 
        # Длина списка равна 1028 согласно охваченному периоду. Используем для этого 
        # pandas period_range, после чего преобразуем список в строковый формат. 
        datas = [str(p) for p in pd.period_range('1940-01-01', periods=1028, freq='M')]
        # В цикле заменим дефис между месяцем и годом на нижнее подчёркивание, 
        # а в конец преобразованной таким образом даты добавим ещё одно, 
        # чтобы отделять дату от названия климатической характеристики. 
        for dat in range(len(datas)):
            # заменяем "-" на "_"
            datas[dat] = re.sub('-', '_', datas[dat])
            # добавляем "_" в конец
            datas[dat] = datas[dat] + '_'


        # создадим список индексов от 1 (именно "1" – так в исходных данных) до 1028
        nums = list(range(1, 1029))
        # В цикле преобразуем их в строковый вид, добавив букву 'b' в начало и 
        # нижнее подчёркивание в конец. Это нужно для замены: в названиях столбцов 
        # исходных данных цифровой индекс предваряется 'b', а после него следует "_". 
        for _ in range(len(nums)):
            nums[_] = 'b' + str(nums[_]) + '_' 

        # для замены в дальнейшем создаём из индексов (nums) и дат (datas) словарь
        datas = dict(zip(nums, datas))

        # Создаём список 'newcols' из заголовков столбцов фрейма 'data_abies'. 
        # При этом исходим из того, что заголовки в 'data_not_abies' те же самые; 
        # это было проверено в одной из ячеек выше. 
        newcols = list(colnames)
        # Сначала создаём цикл, в котором итераторами будут названия столбцов 'data_abies' 
        # ('strings') и соответствующие им цифровые индексы ('index').  
        for index, string in list(enumerate(colnames)): 
            # Во вложенном цикле итераторами будут ключи 
            # ('key'; т.е., исходные обозначения временнЫх периодов) 
            # и значения ('value'; т.е., даты) списка 'datas', подготовленного выше. 
            for key, value in datas.items(): 
                # Если ключ, кодирующий в исходных данных временнОй период, присутствует в 
                # названии столбца, то он заменяется на соответствующую дату методом 'replace', 
                # а результат записывается в соответствующее индексу место списка 'newcols'. 
                if key in string: 
                    newcols[index] = string.replace(key, value)

        # Приводим элементы списка 'newcols' (будущие заголовки столбцов) 
        # к написанию строчными буквами. 
        newcols = [col.lower() for col in newcols]

        # заменяем избыточно длинные названия признаков на более краткие, но информативные
        newcols = [
            elt.replace('relative_humidity', 'rel_hum') for elt in newcols
            ]
        newcols = [
            elt.replace('2m_temperature', '2m_temp') for elt in newcols
            ]
        newcols = [
            elt.replace('total_precipitation', 'pre') for elt in newcols
            ]
        newcols = [
            elt.replace('evaporation', 'evap') for elt in newcols
            ]
        newcols = [
            elt.replace('swvl1', 'soil_water') for elt in newcols
            ]

        # И наконец заменяем названия столбцов обоих фреймов данных 
        # на преобразованные (коды дат заменены на нормальные даты, буквы только строчные). 
        data.columns = pd.Index(newcols)

        # Сделаем индексом первый столбец каждого фрейма с исходными данными, 
        # а затем удалим его. 
        data.index = data['pointid'].astype('int')
        data.drop(columns=['pointid'], inplace=True)

        # создадим фрейм с географическими характеристиками
        data_g = data[geogr_feat]

        """ 
        Эта процедура нужна из-за ошибки добавления данных. В некоторых случаях рассчитанные 
        ряды средних значений при добавлении в pandas data frame конвертируются в NaN. 
        Причину этого я так и не понял, но вынужден был отказаться от прямого добавления 
        рассчитанных данных в столбцы в пользу их добавления в массив numpy. 
        """
        # TODO: Возможно, после сделанных изменений в процедуре загрузки её можно исключить, но пока не проверял.
        """ 
        Создадим временные массивы для добавления средних значений характеристик погоды.
        Количество строк равно таковому во фреймах с загруженными данными, количество 
        столбцов - произведение количества признаков погоды на 12 (месяцев). 
        """
        tmp = np.empty(
            (data.shape[0], len(weather_feat) * len(month_num))
            ).T

        # создадим список для сохранения синтезированных имён погодных переменных
        colnames = list(data_g.columns)
        # создадим итератор
        k = 0

        # Для каждого признака погоды и...
        for w_feat in weather_feat:
            # ...для каждого месяца... 
            for month in month_num:
                # ...рассчитаем среднее значение для каждого объекта в пихтовых...
                tmp[k] = weather_means(
                    data = data, feature = w_feat, month = month
                    )
                # ...создадим новое имя переменной и добавим его в список 'colnames'... 
                colnames.append(str(w_feat + '_' + month))
                # ...и наконец обновим счётчик. 
                k = k + 1
        
        # добавим к географическим характеристикам рассчитанные характеристики погоды
        data_g = pd.concat(
            [data_g, pd.DataFrame(tmp.T, index=data_g.index)], 
            axis=1, 
            ignore_index=True
            )

        # изменяем названия столбцов на такие, которые будут описывать их содержимоое
        data_g.columns = pd.Index(colnames)

        # сохраним как отдельный фрейм координаты точек
        coords = data_g[['point_x', 'point_y']]

        # Оставим только те признаки, которые нужны для работы модели, 
        # и упорядочим их так, как требует модель. 
        data_g = data_g[model.feature_names_in_]
        # получим прогноз для точек, вошедших в текущий чанк
        predictions_chunk = model.predict(data_g)
        # добавим полученные данные в результаты работы модели для всего набора данных
        predictions.extend(predictions_chunk)
        # Если обрабатываем данные для пихты, то во фрейм с признаками 
        # добавляем столбец с единицами (указывает на доминирование пихты). 
        if mode=='abies': 
            data_g['real_veg'] = np.ones(data_g.shape[0])
        # Если обрабатываем данные для других пород, добавляем столбец с нулями 
        # (указывает на отсутствие пихты среди эдификаторов). 
        else:
            data_g['real_veg'] = np.zeros(data_g.shape[0])
        # добавим во фрейм с признаками и данными о лесообразователях результаты работы модели
        data_g['predicted_veg'] = predictions_chunk

        # создадим пустой столбец для оценок результатов работы модели
        data_g['prognosis_res'] = ''
        # задаем список условий (логические маски для всей таблицы)
        conditions = [
            (data_g['real_veg'] == 1) & (data_g['predicted_veg'] == 1),  # TP
            (data_g['real_veg'] == 1) & (data_g['predicted_veg'] == 0),  # FN
            (data_g['real_veg'] == 0) & (data_g['predicted_veg'] == 0),  # TN
            (data_g['real_veg'] == 0) & (data_g['predicted_veg'] == 1)   # FP
        ]
        # задаем соответствующие им значения
        choices = ['TP', 'FN', 'TN', 'FP']
        # применяем махом ко всему датафрейму (без циклов!)
        data_g['prognosis_res'] = np.select(conditions, choices, default='')

        # добавим ко фрейму с данными и результатами работы модели ещё и географические координаты
        data_g = pd.concat([data_g, coords], axis=1)
        
        # если файла 'data.csv' с результатами нет, создаём его на основе 'data_g'
        data_path = os.path.join(ROOT_DIR, 'map_current/data.csv')

        if not os.path.isfile(data_path): 
            data_g.to_csv(data_path)
        # если этот файл уже существует, дописываем в него данные из 'data_g'
        else: 
            data_g.to_csv(
                data_path, 
                mode='a', 
                header=not data_path
                )
    
    return predictions
