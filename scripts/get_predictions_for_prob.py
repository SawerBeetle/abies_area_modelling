import pathlib
from pathlib import Path
import pickle

import json

# загрузить конфиг
with open('C:/Users/user/Yandex.Disk/Важные документы/Исходные данные для статей/Моделирование вспышек/Моделирование ареала пихты/Abies_01/config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# загрузить адрес корневой директории
ROOT_DIR = config['ROOT_DIR']

def load_predictions_for_prob(
        # список путей к загружаемым файлам
        path_list: list[str | pathlib.Path]
        ): 
    
    # создаём пустой список объектов данных
    files = []
    # в цикле для кадого пути загружаем файл
    for file_path in path_list: 
        # Если файл данных существует, ...
        if ((Path(ROOT_DIR) / file_path)).is_file(): 
            # ...выводим сообщение о загрузке и...
            print(f'Загружаем данные из файла {file_path}.')
            # читаем данные, добавляя его в 'files'. 
            with open(Path(ROOT_DIR) / file_path, 'rb') as f: 
                files.append(pickle.load(f))
        # Если файл данных не существует, ... 
        else: 
            # ...выводим сообщение об ошибке и продолжаем выполнение цикла. 
            print(f'Файл {file_path} не существует. ')
            continue

    return files