import glob
import os
import re

from charset_normalizer import from_path
import json
import pandas as pd

# загрузить конфиг
with open('C:/Users/user/Yandex.Disk/Важные документы/Исходные данные для статей/Моделирование вспышек/Моделирование ареала пихты/Abies_01/config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# загрузить имя модели
MODEL_NAME = config['MODEL_NAME']
# загрузить адрес корневой папки
ROOT_DIR = config['ROOT_DIR']

def read_log(
        # корневой каталог
        root=ROOT_DIR, 
        # имя модели
        model=MODEL_NAME, 
        # фрейм с данными для создания карты
        for_map=None, 
        # тестовый набор предикторов
        pred_test=None, 
        # режим работы функции ('map' или 'model')
        mode='map'
        ): 
    # получим путь к папке с логом
    path_to_log_dir = os.path.join(root, 'models', model)
    # Получим путь к самому логу; он имеет вид списка с единственным элементом, 
    # так что напрямую его не используешь
    log = glob.glob(os.path.join(path_to_log_dir, '*.txt'))

    try:
        # откроем лог-файл
        with open(*log, 'r', encoding='cp1251') as f:
            # .strip() убирает пробелы и знаки переноса строки по краям
            opened_log = [line.strip() for line in f if line.strip()]
            # преобразуем записи лога во фрейм с единственным столбцом
            log_as_df = pd.DataFrame(opened_log)
    except: 
        print('Проблемы с чтением лог-файла.')

    # В этом режиме функция возвращает данные, 
    # далее обрабатываемые загруженной моделью, обученной ранее. 
    if mode == 'map': 
        # получим строку, от которой начинается список использованных моделью предикторов
        importance_index = log_as_df[log_as_df[0] == 'Важность использованных предикторов:'].index
        # поскольку он внизу лог-файла, то предикторы и их важность получим с помощью этого среза
        used_predictors = log_as_df.loc[importance_index[0] + 1:, :]
        # разделим 'used_predictors' на два столбца (признаки и их важность) по знаку табуляции '\t'
        used_predictors = used_predictors[0].str.split('\t', expand=True)
        # дадим названия столбцам
        used_predictors.columns = ['feature', 'importance']
        # получим 'for_map' в виде, пригодном для дальнейшей работы, оставив только нужные столбцы
        for_map = for_map[['point_x', 'point_y'] + list(used_predictors['feature']) + ['vegetation']]
        # оставим в 'pred_test' (будет нужен при создании рисунков) только столбцы с нужными предикторами
        pred_test = pred_test[list(used_predictors['feature'])]

        return(used_predictors, for_map, pred_test)
    
    # Этот режим нужен для чтения данных о модели из лог-файла и используется при сравнении моделей 
    # 'COMPARE_MODELS_RESULTS == True'. 
    elif mode == 'model': 
        # получим целевую метрику
        # создадим маску для поиска строки с целевой метрикой
        mask = log_as_df[0].str.contains('Целевая метрика')
        # извлечём имя метрики
        target_metric = log_as_df[mask][0].str.split(': ', expand=True).iloc[0][1]
        # удалим знаки препинания
        target_metric = re.sub(r'[^\w\s]', '', target_metric)

        if target_metric == 'fbeta': 
            # создадим маску для поиска строки со значением beta
            mask = log_as_df[0].str.contains('Значение beta')
            # извлечём beta
            beta = log_as_df[mask][0].str.split(': ', expand=True).iloc[0][1]
            # удалим знаки препинания
            beta = re.sub(r'[^\w\s]', '', beta)
            target_metric = target_metric + '_' + str(beta)

        # создадим маску для поиска первой строки таблицы метрик
        mask = log_as_df[0].str.contains('Dummy')
        # получим её индекс
        metrics_index = log_as_df[mask].index[0]
        # получим таблицу с метриками
        metrics = log_as_df.iloc[metrics_index:(metrics_index + 5), 0]
        # удалим лишние пробелы
        metrics = metrics.str.replace(r'\s+', ' ', regex=True)
        # разобъём массив данных на столбцы
        metrics = metrics.str.split(' ', expand=True)
        # назовём столбцы
        metrics.columns = ['algorythm', 'Target', 'accuracy', 'precision', 'recall']
        # все строки, какие можно, конвертируем в численные значения
        for _ in list(range(metrics.shape[1])):
            try:
                metrics.iloc[_] = pd.to_numeric(metrics.iloc[_])
            except: 
                continue
        # сохраним только строку с максимальной целевой метрикой, а столбец с её значениями уберём
        metrics = metrics[metrics['Target'] == max(metrics['Target'])][['algorythm', 'accuracy', 'precision', 'recall']]

        # получим путь к файлу с результатами прогноза
        res_of_prognosis = glob.glob(os.path.join(path_to_log_dir, '*.csv'))
        try:
            # откроем файл с результатами прогнозирования
            prognosis_data = pd.read_csv(*res_of_prognosis)['vegetation_predicted'] 
        except: 
            print('Проблемы с чтением файла с результатами прогнозирования.')

        return target_metric, metrics, prognosis_data
    
    else: 
        print('Измените значение параметра mode на допустимое.')