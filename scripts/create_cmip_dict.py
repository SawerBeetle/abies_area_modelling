import json

# загрузить конфиг
with open('C:/Users/user/Yandex.Disk/Важные документы/Исходные данные для статей/Моделирование вспышек/Моделирование ареала пихты/Abies_01/config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# адрес корневой папки
ROOT_DIR = config['ROOT_DIR']

from load_projections import load_projections

def create_cmip_dict(
        cmip_dict, 
        file_prec, file_rel_hum, file_snow_depth, file_evap, file_soil_water, snow_density, 
        scenario, 
        prefix_prec, prefix_rel_hum, prefix_snow_depth, prefix_evap, prefix_soil_water, 
        monthes_prec, monthes_rel_hum, monthes_snow_depth, monthes_evap, monthes_soil_water): 
    
    precipitation = load_projections(file_prec, scenario, prefix_prec, monthes_prec)
    cmip_dict['pre_01'] = {model: df * 86.4 for model, df in precipitation['jan'].items()}
    cmip_dict['pre_05'] = {model: df * 86.4 for model, df in precipitation['may'].items()}

    relative_humidity = load_projections(file_rel_hum, scenario, prefix_rel_hum, monthes_rel_hum)
    cmip_dict['rel_hum_09'] = relative_humidity['sep']

    snow_depth = load_projections(file_snow_depth, scenario, prefix_snow_depth, monthes_snow_depth)
    cmip_dict['snow_depth_10'] = {model: df.mul(snow_density, axis=0) for model, df in snow_depth['oct'].items()}

    evaporation = load_projections(file_evap, scenario, prefix_evap, monthes_evap)
    cmip_dict['evap_03'] = {model: df * -86.4 for model, df in evaporation['mar'].items()}
    cmip_dict['evap_04'] = {model: df * -86.4 for model, df in evaporation['apr'].items()}

    soil_water = load_projections(file_soil_water, scenario, prefix_soil_water, monthes_soil_water)
    cmip_dict['soil_water_03'] = {model: df / 70 for model, df in soil_water['mar'].items()}
