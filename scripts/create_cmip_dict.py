import json

# загрузить конфиг
with open('C:/Users/user/Yandex.Disk/Важные документы/Исходные данные для статей/Моделирование вспышек/Моделирование ареала пихты/Abies_01/config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# адрес корневой папки
ROOT_DIR = config['ROOT_DIR']

# загрузим функцию 'load_projections'
from load_projections import load_projections

def create_cmip_dict(
        # Заранее созданный словарь, где отдельные элементы будут заполнены 
        # данными о тех или иных характеристиках погоды. 
        cmip_dict: dict, 
        # Имена файлов с информацией о признаках погоды: prec - осадки, 
        # rel_hum - влажность воздуха, snow_depth - глубина снежного покрова, 
        # evap - эвапорация, soil_water - содежание воды в верхнем слое почвы, 
        # snow_density - плотность снега. 
        file_prec: str, file_rel_hum: str, file_snow_depth: str, file_evap: str, file_soil_water: str, snow_density: str, 
        # траектория SSP
        scenario: str, 
        # префиксы, маркирующие в заголовках столбцов ту или иную характеристику климата
        prefix_prec: str, prefix_rel_hum: str, prefix_snow_depth: str, prefix_evap: str, prefix_soil_water: str, 
        # списки с буквенными кодами месяцев
        monthes_prec: list, monthes_rel_hum: list, monthes_snow_depth: list, monthes_evap: list, monthes_soil_water: list
        ): 

    # ----------------------------------------------------------------------------------------------
    # HARDCODE: в этом блоке отдельные элементы словаря ('pre_01', 'evap_03', etc.) названы 
    # под конкретную модель с конкретным набором предикторов. Если модель и предикторы 
    # поменяются, функцию придётся переписывать. 

    # О методах конвертирования исходных данных в размерность, которая пригодна для работы модели, 
    # см. в ноутбуке 'prediction.ipynb' (раздел 3 "Предварительная обработка данных о будущем климате"). 

    # загружаем данные об осадках
    precipitation = load_projections(file_prec, scenario, prefix_prec, monthes_prec) 
    # конвертируем данные об осадках и записываем в словарь
    # январь
    cmip_dict['pre_01'] = {model: df * 86.4 for model, df in precipitation['jan'].items()}
    # май
    cmip_dict['pre_05'] = {model: df * 86.4 for model, df in precipitation['may'].items()}

    # загружаем данные о влажности воздуха
    relative_humidity = load_projections(file_rel_hum, scenario, prefix_rel_hum, monthes_rel_hum)
    # записываем данные о влажности воздуха в словарь
    cmip_dict['rel_hum_09'] = relative_humidity['sep']

    # загружаем данные о глубине снега
    snow_depth = load_projections(file_snow_depth, scenario, prefix_snow_depth, monthes_snow_depth)
    # конвертируем данные о глубине снега и записываем в словарь
    cmip_dict['snow_depth_10'] = {model: df.mul(snow_density, axis=0) for model, df in snow_depth['oct'].items()}

    # загружаем данные об эвапорации
    evaporation = load_projections(file_evap, scenario, prefix_evap, monthes_evap)
    # конвертируем данные об эвапорации и записываем в словарь
    # март
    cmip_dict['evap_03'] = {model: df * -86.4 for model, df in evaporation['mar'].items()}
    # апрель
    cmip_dict['evap_04'] = {model: df * -86.4 for model, df in evaporation['apr'].items()}

    # загружаем данные о содержании воды в верхнем слое почвы
    soil_water = load_projections(file_soil_water, scenario, prefix_soil_water, monthes_soil_water)
    # конвертируем данные о содержании воды в верхнем слое почвы и записываем в словарь
    cmip_dict['soil_water_03'] = {model: df / 70 for model, df in soil_water['mar'].items()}
    # ----------------------------------------------------------------------------------------------
