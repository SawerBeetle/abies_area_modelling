import re
import sys
import winsound

import json
import pandas as pd

from load_raw_data import load_raw_data

# загрузить конфиг
with open('C:/Users/user/Yandex.Disk/Важные документы/Исходные данные для статей/Моделирование вспышек/Моделирование ареала пихты/Abies_01/config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# загрузить объём данных для обработки
DATA_VOLUME = config['DATA_VOLUME']
if DATA_VOLUME < 100: 
    DATA_VOLUME = 100
# загрузить названия файлов для пихты и других лесообразователей
NAME_ABIES_FILE = config['NAME_ABIES_FILE']
NAME_OTHERSP_FILE = config['NAME_OTHERSP_FILE']
# загрузить адрес корневой папки
ROOT_DIR = config['ROOT_DIR']
# загрузить SEED
SEED = config['SEED']

def load_weather_partially(
        abies_file=NAME_ABIES_FILE, 
        othersp_file=NAME_OTHERSP_FILE, 
        data_volume=DATA_VOLUME, 
        seed=SEED
        ): 
    print(f'Объём данных для загрузки по каждой из двух формаций: {data_volume}.')

    # загрузим данные для пихты
    data_abies = load_raw_data(file_name=abies_file, seed=seed, size=data_volume)
    # загрузим данные для прочих сообществ  
    data_not_abies = load_raw_data(file_name=othersp_file, seed=seed, size=data_volume)

    # проверить полноту данных
    if data_abies.shape[1] == 6200: 
        print('Количество загруженных столбцов данных для пихты совпадает с исходным файлом.')
    else: 
        print('Количество загруженных столбцов данных для пихты НЕ СОВПАДАЕТ с исходным файлом.')
        winsound.Beep(1000, 2000)
        sys.exit()
    # проверка количества столбцов для прочих (непихтовых) сообществ
    if data_not_abies.shape[1] == 6200: 
        print('Количество загруженных столбцов данных для прочих сообществ совпадает с исходным файлом.')
    else: 
        print('Количество загруженных столбцов данных для прочих сообществ НЕ СОВПАДАЕТ с исходным файлом.')
        winsound.Beep(1000, 2000)
        sys.exit()
    if all(data_not_abies.columns == data_abies.columns):
        print('Названия столбцов загруженных наборов данных совпадают.')
    else: 
        print('Названия столбцов загруженных наборов данных НЕ СОВПАДАЮТ.')
        winsound.Beep(1000, 2000)
        sys.exit()

    # проверим, все ли данные являются количественными и конвертируем их в количественные, если нужно
    if len(data_abies.select_dtypes(exclude=['number']).columns) > 0:
        for _ in list(data_abies.columns): 
            data_abies[_] = pd.to_numeric(data_abies[_])
    if len(data_abies.select_dtypes(exclude=['number']).columns) > 0:
        for _ in list(data_abies.columns): 
            data_abies[_] = pd.to_numeric(data_abies[_])

    # сохраним названия столбцов
    colnames = data_abies.columns

    # Приведём названия столбцов во фреймах с исходными данными к единообразному виду. 
    # Во-первых, заменим цифровые коды (b1...1032) на годы и месяцы. Во-вторых, 
    # приведём названия столбцов к snake_case. Естественно, всё это выполняется только 
    # Для начала создадим список дат в формате "год_месяц" начиная с января 1940 года. 
    # Длина списка равна 1032 согласно охваченному периоду. Используем для этого 
    # pandas period_range, после чего преобразуем список в строковый формат. 
    datas = [str(p) for p in pd.period_range('1940-01-01', periods=1032, freq='M')]
    # В цикле заменим дефис между месяцем и годом на нижнее подчёркивание, 
    # а в конец преобразованной таким образом даты добавим ещё одно, 
    # чтобы отделять дату от названия климатической характеристики. 
    for dat in range(len(datas)):
        # заменяем "-" на "_"
        datas[dat] = re.sub('-', '_', datas[dat])
        # добавляем "_" в конец
        datas[dat] = datas[dat] + '_'

    # создадим список индексов от 1 (именно "1" – так в исходных данных) до 1032
    nums = list(range(1, 1032))
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
    # При этом исходим из того, что содержимое столбцов в файлах соответсвует друг другу. 
    data_abies.columns = pd.Index(newcols)
    data_not_abies.columns = pd.Index(newcols)

    # Сделаем индексом первый столбец каждого фрейма с исходными данными, 
    # а затем удалим его. 
    data_abies.index = data_abies['pointid'].astype('int')
    data_abies.drop(columns=['pointid'], inplace=True)
    data_not_abies.index = data_not_abies['pointid'].astype('int')
    data_not_abies.drop(columns=['pointid'], inplace=True)

    # проверим на наличие дубликатов
    if data_abies.shape[0] == data_abies.drop_duplicates().shape[0]: 
        print('Дублирующихся строк в наборе данных по пихте нет.')
    else: 
        print('В наборе данных для пихты есть дублирующиеся строки.')
        winsound.Beep(1000, 2000)
        sys.exit()
    if data_not_abies.shape[0] == data_not_abies.drop_duplicates().shape[0]: 
        print('Дублирующихся строк в наборе данных по другим сообществам нет.')
    else: 
        print('В наборе данных для других сообществ есть дублирующиеся строки.')
        winsound.Beep(1000, 2000)
        sys.exit()

    return data_abies, data_not_abies
    