from itertools import chain, repeat
import os
import pickle
import random
from typing import Literal

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm

# загрузить конфиг
with open('C:/Users/user/Yandex.Disk/Важные документы/Исходные данные для статей/Моделирование вспышек/Моделирование ареала пихты/Abies_01/config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# зерно
SEED = config['SEED']
# корневая папка
ROOT_DIR = config['ROOT_DIR']

def get_predictions_for_scenario(
        # исходные данные для моделирования
        cmip_dict: dict, 
        # список имён признаков (= список столбцов в исходящем фрейме)
        colnames: list, 
        # количество объектов (= количество строк в исходящем фрейме)
        n_items: int, 
        # данные о высотах н.у.м.
        srtm: pd.DataFrame | pd.Series | np.ndarray, 
        # объект с моделью, которая будет давать прогноз
        model, 
        # путь для сохранения/загрузки файла с результатами прогноза
        target_path: str, 
        # количество симуляций
        n_simulations=100, 
        # режим обработки данных
        mode: Literal['year', 'quarter'] = 'year',
        # период, для которого обрабатываются данные
        period: list | tuple | None = None, 
        # имя сценария (нужно для записи в лог)
        scenario: str = 'CMIP 3.4'
        ): 

    ######################################################
    # определим функцию для извлечения нужных симуляций
    def extract_simulations(results): 
        # Извлечём симуляции с минимальным количеством "пихтовых" полигонов (далее минимальная симуляция),
        # с медианным (далее медианная) и с максимальным (далее максимальная). 
        
        # создадим маски извлечения
        # для минимальной
        mask_min = results.sum(axis=0) == results.sum(axis=0).min()
        # для медианной
        mask_median = results.sum(axis=0) == results.sum(axis=0).median()
        # для максимальной
        mask_max = results.sum(axis=0) == results.sum(axis=0).max()

        # извлечём симуляции по маскам
        # минимальная
        sim_min = results.loc[:, mask_min]
        # медианная
        # NOTE: Блок if–else ниже нужен, чтобы избежать ситуации, когда медианное значение 
        # предсказанной доли "пихтовых" полигонов не совпадает ни с одним из реальных. 
        # Если совпало (т.е, хотя бы одно значение в маске True), ... 
        if sum(mask_median) > 0: 
            # ...то медианный прогноз извлекаем прямо по маске. 
            sim_median = results.loc[:, mask_median]
        # Если не совпало (все значения в маске False), ...
        else: 
            # ...то считаем количество "пихтовых" полигонов для каждой симуляции, ...
            favorability = results.sum(axis=0)
            # ...рассчитываем медиану этих значений, ... 
            favorability_median = results.sum(axis=0).median()
            # ...модуль разности между ними и медианой, ...
            difference = abs(favorability - favorability_median)
            # ...находим минимальную разность...
            min_difference = min(difference)
            # ...и уже по ней создаём маску для медианного значения. 
            mask_median = difference == min_difference
            # извлекаем медианное значение (или значения)
            sim_median = results.loc[:, mask_median]
        # максимальная
        sim_max = results.loc[:, mask_max]

        # Если в извлечённых по маскам симуляциях более одного столбца, то
        # извлечём по одному случайному столбцу для дальнейшего анализа уже из них. 
        # для минимальной
        if sim_min.shape[1] > 1:
            # зафиксируем генератор случайных чисел
            random.seed(SEED)
            # получим индекс случайного столбца для извлечения
            random_index_min = random.randint(0, sim_min.shape[1])
            # извлекаем
            sim_min = pd.Series(sim_min.iloc[:, random_index_min])
        # если минимальная симуляция изначально одна, то преобразуем её в Series
        else: 
            sim_min = pd.Series(sim_min.to_numpy().flatten())

        # для медианной
        if sim_median.shape[1] > 1:
            # зафиксируем генератор случайных чисел
            random.seed(SEED)
            # получим индекс случайного столбца для извлечения
            random_index_median = random.randint(0, sim_median.shape[1])
            # извлекаем
            sim_median = pd.Series(sim_median.iloc[:, random_index_median])
        # если медианная симуляция изначально одна, то преобразуем её в Series
        else: 
            sim_median = pd.Series(sim_median.to_numpy().flatten())

        # для максимальной
        if sim_max.shape[1] > 1:
            # зафиксируем генератор случайных чисел
            random.seed(SEED)
            # получим индекс случайного столбца для извлечения
            random_index_max = random.randint(0, sim_max.shape[1])
            # извлекаем
            sim_max = pd.Series(sim_max.iloc[:, random_index_max])
        # если максимальная симуляция изначально одна, то преобразуем её в Series
        else: 
            sim_max = pd.Series(sim_max.to_numpy().flatten())

        return sim_min, sim_median, sim_max
    ######################################################

    ######################################################
    def get_results_of_simulations(sim_results): 
        # рассчитаем минимальную, медианную и максимальную долю благоприятных для пихты полигонов
        # сначала все доли
        proportion_of_favorable = 100 * round(sim_results.sum(axis=0) / sim_results.shape[0], 5)
        # минимальная
        min_favorable = proportion_of_favorable.min()
        # медианная
        median_favorable = proportion_of_favorable.median()
        # максимальная
        max_favorable = proportion_of_favorable.max()

        # Настройка стиля графика
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(8, 5))

        # Построение графика плотности (KDE)
        # fill=True закрасит область под кривой
        sns.kdeplot(
            x=proportion_of_favorable, 
            fill=True, 
            color="forestgreen", 
            alpha=0.5, 
            linewidth=2, 
            clip=(0, 100)
            )

        plot_name = 'density_of_simulation_results_' + scenario + '_' + str(period[0]) + '–' + str(period[1]) + '.jpg'
        plt.title(f"Распределение результатов прогноза для {scenario} в {period[0]}–{period[1]} гг.")
        plt.xlabel("Доля благоприятных участков, %")
        plt.ylabel("Плотность")
        plt.savefig(
            os.path.join(ROOT_DIR, 'future_predictions/density_plots', plot_name), 
            dpi=300,              # Качество картинки
            bbox_inches='tight'   # Обрезка лишних полей
        )
        plt.tight_layout()
        plt.show()

        # создадим путь к логу
        log_path = os.path.join(ROOT_DIR, 'future_predictions', 'log.txt')
        # запишем данные в лог-файл, созданный ранее
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write('------------\n')
            log_file.write(f'Сценарий: {scenario}, период: {period[0]}–{period[1]}: \n')
            log_file.write(f'Минимальная доля полигонов, благоприятных для пихты: {min_favorable:.2f}% \n')
            log_file.write(f'Медианная доля полигонов, благоприятных для пихты: {median_favorable:.2f}% \n')
            log_file.write(f'Максимальная доля полигонов, благоприятных для пихты: {max_favorable:.2f}% \n')
            log_file.write(f'------------\n\n')
    ######################################################
                
    # пробуем загрузить данные из parquet
    if os.path.isfile(os.path.join(ROOT_DIR, target_path)): 
        with open(os.path.join(ROOT_DIR, target_path), 'rb') as f:
            all_sim_results = pickle.load(f)
        print('Загружены ранее созданные данные. \n')
        get_results_of_simulations(all_sim_results)

        return extract_simulations(all_sim_results)
    
    elif mode == 'year': 
        print('Рассчитываем данные... \n')
        # Сначала получаем годы: на первом шаге достаём первый ключ 'cmip_dict', ...
        key0 = list(cmip_dict.keys())[0]
        # ...на втором – достаём первый ключ из первого элемента 'cmip_dict', ...
        key1 = list(cmip_dict[key0].keys())[0]
        # ...и наконец извлекаем из него имена столбцов, т.е., годы. 
        years = list(cmip_dict[key0][key1].columns)

        k = 0
        for year in tqdm(years, desc='Обработка лет: '): 
            # Для каждого признака создадим список длиной n_simulations, 
            # в котором будут содержаться имена моделей, используетмых при n-й симуляции. 
            model_names = dict.fromkeys(cmip_dict.keys())
            for key in model_names.keys(): 
                random.seed(SEED)
                try:
                    model_names[key] = [random.choice(list(cmip_dict[key].keys())) for _ in range(n_simulations)]
                except: 
                    print('Error take plase when a model list constructed.')

            # создадим матрицу для записи результатов симуляций
            try:
                sim_results = pd.DataFrame(np.zeros((n_items, n_simulations)))
            except: 
                print('Error take plase when df for results of simulations constructed.')

            for _ in tqdm(list(range(n_simulations)), desc=f'Обработка симуляции для {year} года: ', leave=False): 
                # создаём фрейм с данными для работы модели
                dataset = pd.DataFrame(np.zeros((n_items, len(colnames))), columns=colnames)

                # создаём фрейм с предикторами для подачи в модель
                for column in colnames:
                    # извлекаем из 'cmip_dict' элемент с данными по нужному признаку
                    feature = cmip_dict[column]
                    # забираем оттуда данные по используемой на данном шаге модели
                    cmip_model = model_names[column][_]
                    # помещаем в создаваемый фрейм 'dataset' данные о погоде по конкретному году
                    dataset[column] = feature[cmip_model][year]

                # добавляем данные о высотах н.у.м.
                dataset['srtm'] = srtm

                # выполняем прогноз
                sim_results[_] = model.predict(dataset)

            # Если это первый прогноз, то...
            if k == 0: 
                # ...на его основе создаём общую таблицу результатов прогноза и...
                all_sim_results = sim_results.copy()
                # ...обновляем счётчик. 
                k += 1
            # Если это второй и более прогноз, то...
            else: 
                # добавляем его результаты в уже существующую таблицу. 
                all_sim_results = pd.concat([all_sim_results, sim_results], axis=1)

        # задаём имена столбцов (годы)
        all_sim_results.columns = list(chain.from_iterable(repeat(year, n_simulations) for year in years))

        # сохраним данные в pickle
        with open(os.path.join(ROOT_DIR, target_path), 'wb') as f:
            pickle.dump(all_sim_results, f)
        print('Данные успешно сохранены.')

        return all_sim_results 

    elif mode == 'quarter': 
        print('Рассчитываем данные... \n')
        # Сначала получаем годы: на первом шаге достаём первый ключ 'cmip_dict', ...
        key0 = list(cmip_dict.keys())[0]
        # ...на втором – достаём первый ключ из первого элемента 'cmip_dict', ...
        key1 = list(cmip_dict[key0].keys())[0]
        # ...и наконец извлекаем из него имена столбцов, т.е., годы. 
        years = list(range(period[0], period[1]))

        k = 0

        # создадим матрицу для записи результатов симуляций
        try:
            sim_results = pd.DataFrame(np.zeros((n_items, n_simulations)))
        except: 
            print('Error take plase when df for results of simulations constructed.')

        # Для каждого признака создадим список длиной n_simulations, 
        # в котором будут содержаться имена моделей, используетмых при n-й симуляции. 
        model_names = dict.fromkeys(cmip_dict.keys())
        for key in model_names.keys(): 
            random.seed(SEED)
            try:
                model_names[key] = [random.choice(list(cmip_dict[key].keys())) for _ in range(n_simulations)]
            except: 
                print('Error take plase when a model list constructed.')
                
        # выполним симуляции
        for _ in tqdm(list(range(n_simulations)), desc=f'Обработка симуляции {k + 1}: ', leave=False): 
            # создаём фрейм с данными для работы модели
            dataset = pd.DataFrame(np.zeros((n_items, len(colnames))), columns=colnames)

            # создаём фрейм с предикторами для подачи в модель
            for column in colnames:
                # извлекаем из 'cmip_dict' элемент с данными по нужному признаку
                feature = cmip_dict[column]
                # забираем оттуда данные по используемой на данном шаге модели
                cmip_model = model_names[column][_]
                # помещаем в создаваемый фрейм 'dataset' данные о погоде по периоду
                dataset[column] = feature[cmip_model][years].T.mean().T

            # добавляем данные о высотах н.у.м.
            dataset['srtm'] = srtm

            # выполняем прогноз
            sim_results[_] = model.predict(dataset)

        # сохраним данные в pickle
        with open(os.path.join(ROOT_DIR, target_path), 'wb') as f:
            pickle.dump(sim_results, f)
        print('Данные успешно сохранены.')

        get_results_of_simulations(sim_results)

        return extract_simulations(sim_results)
        
    else: 
        print('Ошибка выбора режима работы функции.')

