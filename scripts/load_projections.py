import os
import re
import datetime as dt

import json
import pandas as pd
from pyarrow import parquet as pq

# загрузить конфиг
with open('C:/Users/user/Yandex.Disk/Важные документы/Исходные данные для статей/Моделирование вспышек/Моделирование ареала пихты/Abies_01/config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# адрес корневой папки
ROOT_DIR = config['ROOT_DIR']

def load_projections(
        # имя файла с данными о погоде
        file: str, 
        # имя траектории SSP
        scenario: str, 
        # префикс, определяющий признак погоды, по которому будем извлекать данные
        prefix: str, 
        # список из трёхбуквенных аббревиатур месяцев, для которых будем извлекать данные
        monthes: list
        ): 
    # создадим пути для чтения ('projections') и сохранения ('predictions') данных
    future_projections = os.path.join(ROOT_DIR, 'future_projections')
    future_predictions = os.path.join(ROOT_DIR, 'future_predictions')

    # дополним путь к данным названием файла
    path = os.path.join(future_projections, file)
    # создадим путь к логу
    log_path = os.path.join(future_predictions, 'log.txt')

    # Создадим пустое множество для имён моделей климата CMIP, данные по которым 
    # есть для данной траектории SSP ('scenario'). 
    unique_models = set()

    # проверим, существует ли папка для сохранения результатов работы функции, если нет, то создадим 
    if not os.path.isdir(future_projections): 
        os.mkdir(future_projections)

    # выберем читаемое название характеристики климата
    match prefix: 
        case 'pr': 
            feature = 'precipitation'
        case 'hur': 
            feature = 'relative humidity'
        case 'snd': 
            feature = 'snow depth'
        case 'evspsbl': 
            feature = 'evaporation'
        case 'mrsos': 
            feature = 'soil water content'

    # Проверим, существует ли лог-файл, если нет, то создадим его, запишем в него данные  
    # о времени выполнения расчётов, траектории климата и признаке, с которым работаем. 
    if not os.path.isfile(log_path): 
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write('------------\n')
            log_file.write(f'Запись сделана {dt.date.today()} {dt.datetime.now().strftime("%H:%M")}\n')
            log_file.write(f'Сценарий: {'.'.join(scenario)}: \n')
            log_file.write(f'Признак: {feature} \n')
    # Если лог-файл существует, запишем в него данные о времени выполнения расчётов, 
    # траектории климата и признаке, с которым работаем. 
    else: 
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write('------------\n')
            log_file.write(f'Запись сделана {dt.date.today()} {dt.datetime.now().strftime("%H:%M")}\n')
            log_file.write(f'Сценарий: {'.'.join(scenario)}: \n')
            log_file.write(f'Признак: {feature} \n')
      
    # Создадим пустой словарь для данных, отфильтрованных по 
    # климатической траектории, характеристике погоды и месяцу. 
    filtered_data = {}

    # получим имена столбцов из файла с климатической проекцией
    # если данные извлекаются из текстового файла
    if path.endswith('txt'):
        cols = pd.read_csv(path, header=None, nrows=1)
        # конвертируем их в список
        cols = cols.iloc[0].to_list()
    # если данные извлекаются из файла parquet
    else: 
        cols = pq.read_schema(path).names

    # создадим маску для имени сценария (т.е., траектории SSP) 'scenario'
    sc_mask = scenario + '_2'
    # создадим маску для выбора столбцов, в которых записаны данные для сценария 'scenario'
    mask = [col for col in cols if sc_mask in col]

    # Создадим список моделей, по которым есть данные для обрабатываемой траектории SSP ('scenario') 
    # и признака погоды, заданного с помощью 'prefix'. 
    # В цикле для каждого столбца из маски...
    for col in mask: 
        # ...определяем, содержит ли он данные о признаке погоды 'prefix', ... 
        match_col = re.search(rf"{prefix}_(.*?)_ssp", col)
        # ...и если да, то...
        if match_col:
            # ...извлекаем из него имя модели, ...
            model_name = match_col.group(1) 
            # ...которое добавляем в множество с именами моделей 'unique_models'. 
            unique_models.add(model_name) 
            
    # превращаем множество в список
    unique_models_list = list(unique_models)

    # читаем данные о климате из файла
    # если данные в текстовом файле
    if path.endswith('txt'):
        data = pd.read_csv(path, usecols=mask)
    # если данные в файле parquet
    else: 
        data = pd.read_parquet(path, columns=mask)

    # извлекаем данные для нужного месяца (месяцев), признака погоды и сценария
    for _ in list(range(0, len(monthes))): 
        # записываем месяц в лог
        with open(log_path, 'a', encoding="utf-8") as log_file:
            log_file.write(f'{monthes[_]}: ')
        # Начинаем заполнять выше созданный словарь с данными о климате; 
        # сейчас создаём пустой словарь для данных по климату обрабатываемого месяца. 
        filtered_data[monthes[_]] = {}
        # создаём счётчик
        k = 1
        # для каждой модели, для которой есть данные по данным сценарию и месяцу
        for model in unique_models_list: 
            # создаём ключ словаря
            key = scenario + '_' + model + '_' + monthes[_]
            # Создаём маску для фильтрации данных, 
            # оставляя только данные для обрабатываемой модели CMIP ('model'). 
            model_mask = [col for col in mask if model in col]
            # получаем номера столбцов, в которых находятся данные по нужному месяцу
            # NB: подразумевается, что месяцы идут по порядку
            cols_for_month = range(_, len(model_mask), len(monthes))
            # оставляем в маске только те столбцы, где лежат данные по нужному месяцу
            model_mask = [model_mask[i] for i in cols_for_month]
            # фильтруем данные по 'model_mask' и записываем их в словарь
            filtered_data[monthes[_]][key] = data[model_mask].copy()
            # переименуем столбцы годами
            try: 
                # HARDCODE: изучаемый здесь наборе данных содержит информацию с 2015 по 2100 гг.
                filtered_data[monthes[_]][key].columns = list(range(2015, 2101))
            # если временной ряд недостаточно длинный
            except ValueError: 
                # удалим данные, в которых не хватает наблюдений
                del filtered_data[monthes[_]][key]
                print(f'Ошибка при обработке признака {feature} для месяца {monthes[_]} при сценарии {model}. ')
            # записываем модель в лог
            with open(log_path, "a", encoding="utf-8") as log_file:
                if k < len(unique_models_list): 
                    log_file.write(f'{model}, ') 
                    k += 1
                else: 
                    log_file.write(f'{model}; \n')
    with open(log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f'------------\n\n')

    # for model in unique_models: 
    return(filtered_data)
