import os

import json
import numpy as np
import pandas as pd

from weather_means import weather_means

# загрузим конфиг
with open('C:/Users/user/Yandex.Disk/Важные документы/Исходные данные для статей/Моделирование вспышек/Моделирование ареала пихты/Abies_01/config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# список характеристик ландшафта
GEOGRAPHICAL_FEATURES = config['GEOGRAPHICAL_FEATURES']
# список признаков погоды
WEATHER_FEATURES = config['WEATHER_FEATURES']
# список номеров месяцев
MONTH_NUMBERS = config['MONTH_NUMBERS']

def calculate_weather_data(
        # характеристики участков с пихтовыми древостоями
        data_abies, 
        # характеристики участков с лесами с господством других пород
        data_not_abies, 
        # характеристики ландшафта
        geogr_feat=GEOGRAPHICAL_FEATURES, 
        # признаки погоды
        weather_feat=WEATHER_FEATURES, 
        # номера месяцев
        month_num=MONTH_NUMBERS 
        ): 
    # Создадим обобщённые (generalized; суффикс 'g') наборы данных для пихты и прочих сообществ. 
    # Туда занесём данные о рельефе, координатах и, далее, усреднённые данные о погодных условиях.
    data_abies_g = data_abies[geogr_feat]
    data_not_abies_g = data_not_abies[geogr_feat]

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
    abies_tmp = np.empty(
        (data_abies.shape[0], len(weather_feat) * len(month_num))
        ).T
    not_abies_tmp = np.empty(
        (data_not_abies.shape[0], len(weather_feat) * len(month_num))
        ).T
    # создадим список для сохранения синтезированных имён погодных переменных
    colnames = list(data_abies_g.columns)
    # создадим итератор
    k = 0
    # Для каждого признака погоды и...
    for w_feat in weather_feat:
        # ...для каждого месяца... 
        for month in month_num:
            # ...рассчитаем среднее значение для каждого объекта в пихтовых...
            abies_tmp[k] = weather_means(
                data = data_abies, feature = w_feat, month = month
                )
            # ...и прочих участках, ...
            not_abies_tmp[k] = weather_means(
                data = data_not_abies, feature = w_feat, month = month
                )
            # ...создадим новое имя переменной и добавим его в список 'colnames'... 
            colnames.append(str(w_feat + '_' + month))
            # ...и наконец обновим счётчик. 
            k = k + 1

    # создадим окончательные фреймы со среднегодовыми значениями характеристик погоды
    # для пихты
    data_abies_g = pd.concat(
        [data_abies_g, pd.DataFrame(abies_tmp.T, index=data_abies_g.index)], 
        axis=1
        )
    # для других пород деревьев
    data_not_abies_g = pd.concat(
        [data_not_abies_g, pd.DataFrame(not_abies_tmp.T, index=data_not_abies_g.index)], 
        axis=1
        )
    # изменяем названия столбцов на такие, которые будут описывать их содержимоое
    data_abies_g.columns = pd.Index(colnames)
    data_not_abies_g.columns = pd.Index(colnames)

    return data_abies_g, data_not_abies_g