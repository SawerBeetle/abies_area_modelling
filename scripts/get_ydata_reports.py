import os

import json
import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport

# загрузить конфиг
with open('C:/Users/user/Yandex.Disk/Важные документы/Исходные данные для статей/Моделирование вспышек/Моделирование ареала пихты/Abies_01/config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# адрес корневой папки
ROOT_DIR = config['ROOT_DIR']
# параметры отчёта ydata
COMPARISON_REPORT_PARAMS = config['YDATA_REPORT_PARAMS']

# Функция создаёт отчёты по разведочному анализу, сравнивая 
#     - с одной стороны данные об историческом климате в пихтовых и прочих полигонах, 
#     - с другой данные о будущем климате для выбранного сценария CMIP. 
# Данные по каждому показателю будущего климата усредняются 
#     - для каждого года и 
#     - по всем моделям, для которых этот признак климата имеется для изучаемого сценария. 
# В результате мы получаем усреднённые данные о будущем климате для каждого полигона. 

def get_ydata_reports(
        data_future: dict, 
        altitudes: pd.Series, 
        columns_order: list[str], 
        title: str
        ): 
    # HARDCODE: предполагается, что данные для пихты и прочих пород берутся 
    # именно из этих файлов и именно из этого каталога. 
    # Загрузка данных для пихты, фильтрация столбцов и установка их в нужном порядке 
    # с помощью 'column_order'. 
    data_abies = pd.read_csv(os.path.join(ROOT_DIR, 'data/abies.csv'))[columns_order]
    # то же, для прочих пород
    data_not_abies = pd.read_csv(os.path.join(ROOT_DIR, 'data/not_abies.csv'))[columns_order]

    # Создаём словарь для данных о климате, смоделированном в рамках изучаемой модели CMIP. 
    # Каждый ключ словаря – характеристика погоды. 
    future_data_to_ydata = dict.fromkeys(data_future.keys())
    # в цикле заполняем словарь 'future_data_to_ydata'
    for key in data_future.keys(): 
        # В список 'data' сохраняем данные всех моделей, которые имеются для 
        # конкретного признака погоды 'key'. 
        # Элементы списка являются фреймами Pandas. 
        data = list(data_future[key].values())
        # Для списка 'data' рассчитываем среднее значение характеристики погоды 'key' 
        # для всех моделей и каждого года. 
        mean_data = np.nanmean([df.values for df in data], axis=0)
        # конвертируем массив numpy 'mean_data' во фрейм 
        future_data_to_ydata[key] = pd.DataFrame(
            mean_data, index=data[0].index, columns=data[0].columns
            )
        # Рассчитаем среднемноголетнее значение признака погоды 'key' 
        # для каждого полигона (строки). 
        future_data_to_ydata[key] = future_data_to_ydata[key].mean(axis=1)#.T.mean().T
    # Конвертируем словарь со среднемноголетними (и усреднёнными по всем моделям) 
    # значениями характеристик будущего климата из словаря во фрейм. 
    future_data_to_ydata = pd.DataFrame(future_data_to_ydata)
    # добавим данные о высотах над уровнем моря
    future_data_to_ydata['srtm'] = altitudes.values
    # на всякий случай приведём типы данных к количественным
    future_data_to_ydata = future_data_to_ydata.apply(pd.to_numeric, errors='coerce')

    # HACK: добавление шума нужно для того, чтобы ydata-profiling отображал графики 
    # при малом количестве уникальных значений. 
    # Этот эффект имел место при анализе относительной влажности сентября (rel_hum_09) 
    # в сценарии CMIP3.4; если это будет повторяться для других характеристик, 
    # нужно применить добавление шума ко всем столбцам, кроме 'srtm'. 
    noise = np.random.uniform(-1e-7, 1e-7, size=len(future_data_to_ydata))
    future_data_to_ydata['rel_hum_09'] = future_data_to_ydata['rel_hum_09'] + noise

    # создадим отчёт об историческом климате для полигонов с преобладанием пихты
    report_abies = ProfileReport(
        data_abies, 
        title="Abies", 
        # type_schema=forced_types,
        **COMPARISON_REPORT_PARAMS
        )
    # то же, для полигонов с преобладанием прочих пород
    report_not_abies = ProfileReport(
        data_not_abies, 
        title="Other species", 
        **COMPARISON_REPORT_PARAMS
        )
    # то же, для усреднённых данных о будущем климате
    future_data_to_ydata = ProfileReport(
        future_data_to_ydata, 
        title=title, 
        **COMPARISON_REPORT_PARAMS
        )

    # сравним прогнозные климатические данные с данными для пихтовых полигонов
    comparison_report_abies = future_data_to_ydata.compare(report_abies)
    # выведем результат сравнения на экран
    comparison_report_abies.to_notebook_iframe()
    # сохраним результаты на диск
    file_title = 'current_abies_vs_' + title + '.html'
    comparison_report_abies.to_file(
        os.path.join(ROOT_DIR, 'eda', file_title)
    )

    # те же действия для полигонов с преобладанием прочих пород
    comparison_report_other = future_data_to_ydata.compare(report_not_abies)
    comparison_report_other.to_notebook_iframe()
    file_title = 'current_other_species_vs_' + title + '.html'
    comparison_report_other.to_file(
        os.path.join(ROOT_DIR, 'eda', file_title)
    )
     