import os
import pickle

import  json
import pandas as pd

# загрузить конфиг
with open('C:/Users/user/Yandex.Disk/Важные документы/Исходные данные для статей/Моделирование вспышек/Моделирование ареала пихты/Abies_01/config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# загрузить коэффициент бета
BETA = config['BETA']
# загрузить количество проходов Optuna
N_TRIALS = config['N_TRIALS']
# загрузить адрес корневой папки
ROOT_DIR = config['ROOT_DIR']
# загрузить целевую метрику
TARGET_METRIC = config['TARGET_METRIC']

def save_best_model(
        # лучшая модель
        model_best, 
        # метка даты и времени для создания имени модели
        dt, 
        # путь к рабочей папке
        path_to_folder, 
        # имя лог-файла
        fname, 
        # алгоритм работы лучшей модели
        best_algorythm, 
        # лучшее значение целевой метрики на валидации
        t_metric_best, 
        # лучшее значение целевой метрики на тестовой выборке
        t_metric_test, 
        # фрейм с метриками (accuracy, precision, recall) лучшей модели
        best_metrics, 
        # предикторы, используемые в работе модели
        used_predictors
        ): 
    # зададим путь к рабочей папке
    os.chdir(path_to_folder)
    # создадим название лучшей модели, включающее дату и время создания
    model_name = 'abies_area_model_' + dt + '_' + TARGET_METRIC
    # сохраним лучшую модель
    pickle.dump(model_best, open(model_name, 'wb')) 

    # запишем данные о модели в лог
    try:
        with open(fname, 'a') as file: 
            file.write('\n')
            # целевая метрика
            file.write(f'Целевая метрика: {TARGET_METRIC}. \n')
            # значение коэффициента fbeta, если при обучении использовали Fbeta как целевую
            if TARGET_METRIC == 'fbeta': 
                file.write(f'    Значение beta: {BETA}. \n')
            # количество проходов обучения
            file.write(f'Лучшая модель обучена за {N_TRIALS} проходов. \n')
            # алгоритм работы лучшей модели
            file.write(f'Алгоритм лучшей модели: {best_algorythm}. \n')
            # целевая метрика лучшей модели при валидации
            file.write(f'    {TARGET_METRIC} лучшей модели (валидация): {t_metric_best}. \n')
            # целевая метрика лучшей модели на тестовом наборе
            file.write(f'    {TARGET_METRIC} лучшей модели (тест): {t_metric_test}. \n\n')
            # таблица с прочими метриками
            file.write(best_metrics.to_string())
            file.write('\n\n    параметры лучшей модели: \n')
    except:
        print('Проблема с логированем данных лучшей модели.')

    # запишем гиперпараметры лучшей модели в лог в форме фрейма
    pd.DataFrame.from_dict(
        model_best.get_params(), orient='index'
        ).to_csv(
            (os.getcwd() + '/' + fname), sep='\t', header=False, mode='a'
        )
    
    # подготовим лог к записи данных об использованных предикторах и их importance
    try:
        with open(fname, 'a') as file: 
            file.write('\n')
            file.write('Важность использованных предикторов: \n')
    except:
        print('Проблема с логированем данных о предикторах.')

    # сохраним в лог-файл таблицу importances
    used_predictors.to_csv(
        (os.getcwd() + '/' + fname), sep='\t', index=False, header=False, mode='a'
        )        