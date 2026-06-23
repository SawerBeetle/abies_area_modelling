import os
import mmap

import io 
import json
import numpy as np
import pandas as pd 
from tqdm.auto import tqdm

# загрузить конфиг
with open('C:/Users/user/Yandex.Disk/Важные документы/Исходные данные для статей/Моделирование вспышек/Моделирование ареала пихты/Abies_01/config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# загрузим объём данных, нужный для обучения моделей
DATA_VOLUME = config['DATA_VOLUME']
# адрес корневой папки
ROOT_DIR = config['ROOT_DIR']
# SEED
SEED = config['SEED']

def load_raw_data(
        # название файла с данными о погоде
        file_name, 
        # затравка для генератора случайных чисел
        seed=SEED, 
        # количество строк, которое нужно загрузить
        size=DATA_VOLUME
        ):
    # создадим путь к файлу с загружаемыми данными
    filepath = os.path.join(ROOT_DIR, "data", file_name)

    # открываем файл для построения карты смещений и чтения
    with open(filepath, "r+b") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:

            # собираем карту индексов строк (индекс -> позиция начала строки в байтах)
            line_offsets = [0]
            # читаем строки; указатель останавливается в начале этих строк
            while m.readline():
                # добавляем позицию указателя с помощью 'tell()' в список с индексами строк
                line_offsets.append(m.tell())

            # рассчитываем количество строк в файле
            n_lines = len(line_offsets)

            # генерируем и сортируем случайные индексы строк
            rng = np.random.default_rng(seed=seed)
            # индексы от 1 до n_lines-1 (0-я строка — это заголовок)
            lines_to_extract = rng.choice(n_lines - 1, size=size, replace=False) + 1
            lines_sorted = sorted(lines_to_extract)

            # извлекаем строку заголовка (индекс 0)
            m.seek(line_offsets[0])
            # Читаем строку, удаляем ненужные символы с концов ('strip') 
            # и декодируем (лог создан в utf-8). 
            header_str = m.readline().strip().decode("utf-8")
            # приводим записи в столбце к читаемому виду
            header_cols = (
                pd.read_csv(io.StringIO(header_str), sep=";", header=None).
                iloc[0].
                str.
                lower()
            )

            # последовательно считываем нужные строки без перезапуска файла
            # создаём пустой список для строк
            lines_list = []
            # читаем строки с номерами, ранее сохранёнными в 'lines_sorted'
            for line_idx in tqdm(
                lines_sorted, desc=f"Загрузка данных из {file_name}"
            ):
                # Переходим к байтовому смещению нужной строки: 
                # с помощью 'seek()' перемещаем указатель в начало строки с номером 'line_idx'. 
                m.seek(line_offsets[line_idx])
                # Читаем, очищаем и декодируем байты в текст
                # читаем и декодируем строку
                line_str = m.readline().strip().decode("utf-8")
                # добавляем прочитанную строку в 'lines_list'
                lines_list.append(line_str)

    # собираем финальный DataFrame за один проход
    full_text = "\n".join(lines_list)
    # здесь 'read_csv' узнаёт конец строки по скрытому символу '\n', остальное ясно
    raw_data = pd.read_csv(io.StringIO(full_text), sep=";", decimal=',', header=None)
    raw_data.columns = header_cols

    return raw_data