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
        # HARDCODE: подразумевается, что высота н.у.м. – единственная характеристика рельефа, 
        # используемая как предиктор. Если набор предикторов сменится, и в него будут включены 
        # другие характеристики рельефа, код придётся переписывать. 
        srtm: pd.DataFrame | pd.Series | np.ndarray, 
        # объект с моделью, которая будет давать прогноз
        model, 
        # путь для сохранения/загрузки файла с результатами прогноза
        target_path: str, 
        # количество симуляций
        n_simulations=100, 
        # режим обработки данных ('year' - обрабатывается каждый год, 'quarter' - 25-летние отрезки)
        mode: Literal['year', 'quarter'] = 'year', 
        # режим выдачи данных (mmm – минимум/медиана/максимум, all – все результаты)
        obtain: Literal['mmm', 'all'] = 'mmm', 
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

        # возвращаем результаты; позже они понадобятся для создания одной из карт
        return sim_min, sim_median, sim_max
    ######################################################

    ######################################################
    # Определим функцию для создания графиков плотности вероятности доли участков, 
    # благоприятствующих росту пихтовых древостоев, и записи в лог-файл минимальной, медианной 
    # и максимальной доли благоприятных полигонов, полученных в ходе симуляций. 

    def get_results_of_simulations(
            # фрейм с результатами симуляций (строки – пиксели, столбцы – симуляции)
            sim_results
            ): 
        # Рассчитаем доли благоприятных для пихты полигонов по всем симуляциям, 
        # затем извлечём минимальную, медианную и максимальную долю. 
        # сначала все доли
        proportion_of_favorable = 100 * round(sim_results.sum(axis=0) / sim_results.shape[0], 5)
        # минимальная
        min_favorable = proportion_of_favorable.min()
        # медианная
        median_favorable = proportion_of_favorable.median()
        # максимальная
        max_favorable = proportion_of_favorable.max()

        # настройка стиля графика (белый фон, серая сетка)
        sns.set_theme(style="whitegrid")
        # зададим размеры графика
        plt.figure(figsize=(8, 5))

        # построение графика плотности (KDE); 'fill=True' закрасит область под кривой
        sns.kdeplot(
            x=proportion_of_favorable, 
            fill=True, 
            color="forestgreen", 
            alpha=0.5, 
            linewidth=2, 
            clip=(0, 100)
            )

        # сформируем имя файлв для сохранения графика KDE
        plot_name = 'density_of_simulation_results_' + scenario + '_' + str(period[0]) + '–' + str(period[1]) + '.jpg'
        # создадим заголовок графика
        plt.title(f"Empirical distribution of prognosis results for {scenario} in {period[0]}–{period[1]} гг.")
        # создадим подпись оси абсцисс
        plt.xlabel("Proportion of the favorable polygons, %")
        # создадим подпись оси ординат
        plt.ylabel("Probability density")
        # сохраним график на диск
        plt.savefig(
            os.path.join(ROOT_DIR, 'future_predictions/density_plots', plot_name), 
            dpi=300, 
            # обрезка лишних полей
            bbox_inches='tight'
        )
        # ещё раз обрежем поля
        plt.tight_layout()
        # выведем график на экран
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

    ######################################################
    # Определим функцию, выводящую на экран таблицу с распрепделением по децилям суммарных  
    # по всем симуляциям вероятностей быть благоприятным для пихты каждого пикселя. 
    # Отдельно выводится доля пикселей с полностью нулевой вероятностью. 
    def count_intervals(
            # фрейм с результатами симуляций (строки – пиксели, столбцы – симуляции)
            sim_results
            ): 
        # Сумма по строкам. Т.к. благоприятный прогноз обозначается как 1, то 
        # эта сумма, по сути, процент вероятности конкретного пикселя быть благоприятным 
        # для роста пихты. 
        sim_sums = sim_results.sum(axis=1)

        # задаём границы интервалов (децилей)
        intervals = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        # создаём пустой список, который затем заполним долями пикселей в пределах каждого дециля
        tiles_in_intervals = []
        # создаём пустой список для записи границ децилей (в формате (low; end]))
        indices = []
        # рассчитываем общее количество пикселей
        no_tiles = sim_sums.count()
        # переходим к расчёту долей пикселей в каждом дециле
        for limit in intervals: 
            # Если обрабатываемое значение равно нулю, то подсчитываем, 
            # сколько пикселей на всех симуляциях имели нулевую вероятность 
            # появления благоприятных условий для пихтовых древостоев. 
            if limit == 0: 
                # добавляем в 'tiles_in_intervals' количество пикселей, у которых в sim_sums нулевое значение
                tiles_in_intervals.append(len(list(filter(lambda x: x == limit, sim_sums))))
                # добавляем в список с границами соответствующую запись (от нуля до нуля включительно)
                indices.append([0.00, 0.00])
            # Если обрабатываемое значение больше нуля, то рассматриваем его как верхнюю границу дециля 
            # (включительно), а предыдущее – как нижнюю границу (исключительно). 
            else: 
                # подсчитываем количество пикселей, для которых значение в 'sim_sums' попадает в нужный интервал
                current_interval_count = len(list(filter(lambda x: (x <= limit) and (x > (limit - 10)), sim_sums)))
                # добавляем это значение в 'tiles_in_intervals'
                tiles_in_intervals.append(current_interval_count)
                # добавляем в список с границами запись, которая эти границы описывает
                indices.append(str(f"({limit - 10:.1f}, {limit:.1f}]"))

        # конвертируем 'tiles_in_intervals' в pd.Series, форматируя данные в нём как проценты
        tiles_in_intervals = pd.Series(tiles_in_intervals / no_tiles).map("{:.2%}".format)
        # конвертируем 'indices' в pd.Series
        indices = pd.Series(indices)
        # Создаём фрейм с информацией о том, как вероятность 
        # благоприятных условий для пихтовых древостоев распределена по децилям. 
        output_df = pd.DataFrame({
            'intervals': indices, 
            'percentage of points': tiles_in_intervals
        })
        # выводим фрейм на экран
        display(output_df)
    ######################################################
                
    # пробуем загрузить данные из parquet
    if os.path.isfile(os.path.join(ROOT_DIR, target_path)): 
        with open(os.path.join(ROOT_DIR, target_path), 'rb') as f:
            all_sim_results = pickle.load(f)
        print('Загружены ранее созданные данные. \n')
        # выводим на экран график KDE
        get_results_of_simulations(all_sim_results)
        # Если функция работает в режиме исследования 25-летних отрезков, ... 
        if mode == 'quarter':
            # ...выводим на экран таблицу с распределением по децилям вероятностей наступления 
            # благоприятных условий для пихтовых древостоев на экран. 
            count_intervals(all_sim_results)

        # Извлечём результаты для симуляции с минимальным предсказанным значением 
        # пикселей, благоприятных для пихты, с медианным и максимальным значением. 
        return extract_simulations(all_sim_results)

    # если работаем в режиме 'year', т.е., выполняем симуляции для каждого года от 2025 до 2100
    elif mode == 'year': 
        print('Рассчитываем данные... \n')
        # Сначала получаем годы: на первом шаге достаём первый ключ 'cmip_dict' (признак погоды), ...
        key0 = list(cmip_dict.keys())[0]
        # ...на втором – достаём первый ключ из первого элемента 'cmip_dict' (модель CMIP), ...
        key1 = list(cmip_dict[key0].keys())[0]
        # ...и наконец извлекаем из него имена столбцов, т.е., годы. 
        years = list(cmip_dict[key0][key1].columns)

        # инициализируем счётчик
        k = 0
        # выполняем симуляции для каждого года
        for year in tqdm(years, desc='Обработка лет: '): 
            # Для каждого признака погоды создадим список длиной n_simulations, 
            # в котором будут содержаться имена моделей, используетмых при n-й симуляции. 
            # сначала создаём словарь, где ключи – имена признаков погоды
            model_names = dict.fromkeys(cmip_dict.keys())
            # далее для каждого ключа в словаре добавляем имена моделей
            for key in model_names.keys(): 
                # устанавливаем зерно генератора
                random.seed(SEED)
                # Создаём список из n_simulations имён моделей для каждого элемента погоды. 
                # Выбираем имена из ключей 'cmip_dict', которые вложены в словарь для 
                # элемента погоды 'key'. 
                try:
                    model_names[key] = [random.choice(list(cmip_dict[key].keys())) for _ in range(n_simulations)]
                # если что-то пошло не так, выводим сообщение об ошибке
                except: 
                    print('Error take plase when a model list constructed.')

            # создадим матрицу для записи результатов симуляций
            try:
                sim_results = pd.DataFrame(np.zeros((n_items, n_simulations)))
            # если что-то пошло не так, выводим сообщение об ошибке
            except: 
                print('Error take plase when df for results of simulations constructed.')

            # создаём симуляции для года 'year' (см. голову цикла)
            for _ in tqdm(list(range(n_simulations)), desc=f'Обработка симуляции для {year} года: ', leave=False): 
                # создаём фрейм для данных для работы модели (пока заполненный нулями)
                dataset = pd.DataFrame(np.zeros((n_items, len(colnames))), columns=colnames)

                # создаём фрейм с предикторами для подачи в модель
                # 'column' здесь – имя обрабатываемой характеристики погоды
                for column in colnames:
                    # извлекаем из 'cmip_dict' элемент с данными по нужному признаку
                    feature = cmip_dict[column]
                    # забираем оттуда данные по используемой на данном шаге модели
                    cmip_model = model_names[column][_]
                    # Помещаем во фрейм 'dataset' данные о погоде по признаку 'feature' и конкретному году 
                    # по данным модели номер '_'. 
                    dataset[column] = feature[cmip_model][year]

                # добавляем данные о высотах н.у.м.
                dataset['srtm'] = srtm

                # выполняем прогноз и добавляем его в текущий столбец матрицы результатов
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

    # если работаем в режиме 'quarter', т.е., выполняем симуляции для каждого 25-летних периодов
    elif mode == 'quarter': 
        print('Рассчитываем данные... \n')
        # Сначала получаем годы: на первом шаге достаём первый ключ 'cmip_dict', ...
        key0 = list(cmip_dict.keys())[0]
        # ...на втором – достаём первый ключ из первого элемента 'cmip_dict', ...
        key1 = list(cmip_dict[key0].keys())[0]
        # ...и наконец извлекаем из него имена столбцов, т.е., годы. 
        years = list(range(period[0], period[1]))

        # инициализируем счётчик
        k = 0

        # создадим матрицу для записи результатов симуляций
        try:
            sim_results = pd.DataFrame(np.zeros((n_items, n_simulations)))
        # если что-то пошло не так, выводим сообщение об ошибке
        except: 
            print('Error take plase when df for results of simulations constructed.')

        # Для каждого признака создадим список длиной n_simulations, 
        # в котором будут содержаться имена моделей, используетмых при n-й симуляции. 
        # сначала создаём словарь, где ключи – имена признаков погоды
        model_names = dict.fromkeys(cmip_dict.keys())
        # далее для каждого ключа в словаре добавляем имена моделей
        for key in model_names.keys(): 
            # устанавливаем зерно генератора
            random.seed(SEED)
            # Создаём список из n_simulations имён моделей для каждого элемента погоды. 
            # Выбираем имена из ключей 'cmip_dict', которые вложены в словарь для 
            # элемента погоды 'key'. 
            try:
                model_names[key] = [random.choice(list(cmip_dict[key].keys())) for _ in range(n_simulations)]
            # если что-то пошло не так, выводим сообщение об ошибке
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

            # выполняем прогноз и добавляем его в текущий столбец матрицы результатов
            sim_results[_] = model.predict(dataset)

        # сохраним данные в pickle
        with open(os.path.join(ROOT_DIR, target_path), 'wb') as f:
            pickle.dump(sim_results, f)
        print('Данные успешно сохранены.')

        # В зависимости от режима выдачи данных либо...
        # ...возвращаем минимальный, медианный и максимальный сценарии, либо...
        if obtain == 'mmm': 
            get_results_of_simulations(sim_results)
            return extract_simulations(sim_results)
        # ...возвращаем результаты всех симуляций. 
        elif obtain == 'all':
            return sim_results
        # на случай выбора ошибочного значения obtain предусмотрено сообщение об этом
        else: 
            print('Ошибка выбора режима выдачи данных (нужно "mmm" или "all".)')

    # если выбрано неверное значение 'mode' (любое кроме 'year' и 'quarter'), сообщим об этом
    else: 
        print('Ошибка выбора режима работы функции.')

