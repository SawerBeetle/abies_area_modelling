import os
import string

import json
import matplotlib.pyplot as plt
import math
import seaborn as sns

# загрузить конфиг
with open('C:/Users/user/Yandex.Disk/Важные документы/Исходные данные для статей/Моделирование вспышек/Моделирование ареала пихты/Abies_01/config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# загрузить путь к корневой папке
ROOT_DIR = config['ROOT_DIR']

def plot_future_climate(cmip_dict: dict, target_path: str, n_cols=3):

    # извлечём ключи (сценарии SSP) из обрабатываемого слованя
    keys0 = list(cmip_dict.keys())

    # зададим ширину графика
    width = 20.5
    # рассчитаем нужное количество строк и высоту графика
    n_rows = math.ceil(len(cmip_dict) / n_cols)
    height = 3 * n_rows

    # создадим полотно для графиков
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(width, height))
    # сделаем нумерацию субграфиков одномерной
    ax_flat = ax.flatten()

    # определим счётчик
    k = 0
    # получим список заглавных букв для заголовков субграфиков
    en_letters = list(string.ascii_uppercase)

    # В цикле создадим и добавим на полотно субграфики, 
    # последовательно обработав сценарии SSP из словаря 'cmip_dict'. 
    for key0, axis in zip(keys0, ax_flat): 
        # извлечём из словаря для обрабатываемого сценария ключи, содержащие имена моделей
        keys1 = list(cmip_dict[key0])
        # последовательно обработаем модели, данные для которых находятся в 'cmip_dict[key0]'
        for key1 in keys1: 
            # Получаем имя модели: находим индекс первого и последнего подчеркивания...
            first_idx = key1.find("_")
            last_idx = key1.rfind("_")
            # ...и вырезаем часть между ними 
            # (прибавляем 1, чтобы не включать первое подчеркивание). 
            model_name = key1[first_idx + 1 : last_idx]

            # создадим линейный график
            sns.lineplot(
                # данные для графика
                data=cmip_dict[key0][key1].mean(axis=0), 
                # субграфик, на который попадёт построенный график
                ax=axis, 
                lw=.5, 
                label=model_name
                )
        # добавляем буквы
        axis.set_title(
            label=en_letters[k], 
            loc='left',          # прижимает текст к левому краю
            fontweight='bold', 
            fontsize=14
        )
        axis.set_xlabel('Year')
        axis.set_ylabel(key0)
        axis.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=8, frameon=True)
        # обновим счётчик
        k += 1
    
    # предотвратим наложение субграфиков
    plt.tight_layout()
    # сохраним графики в папку с моделью
    plt.savefig(os.path.join(ROOT_DIR, target_path))
    # выведем графики на экран
    plt.show()


