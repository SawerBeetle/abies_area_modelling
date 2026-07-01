import json
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier 
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, fbeta_score  
from sklearn.utils import shuffle

# загрузить конфиг
with open('C:/Users/user/Yandex.Disk/Важные документы/Исходные данные для статей/Моделирование вспышек/Моделирование ареала пихты/Abies_01/config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# загрузить коэффициент бета
BETA = config['BETA']
# загрузить количество проходов Optuna
N_TRIALS = config['N_TRIALS']
# загрузить SEED
SEED = config['SEED']

def optimize_model(
        pred_train, 
        veg_train, 
        pred_valid, 
        veg_valid, 
        t_metric_dummy, 
        best_metrics, 
        best_retained_columns, 
        parameters, 
        model=RandomForestClassifier, 
        target_metric='roc_auc', 
        index='RF', 
        algorythm='random forest', 
        study_name='rf', 
        t_metric_best=0, 
        model_best=None, 
        best_algorythm=None
        ): 
    '''
    Оптимизация набора предикторов через удаление тех из них, 
    которые имеют слишком низкую (менее средней) важность. 
    '''
    # зафиксируем исходное количество признаков-предикторов
    default_featurs_num = pred_train.shape[1]

    # проверим, можно ли задать SEED, и создадим экземпляр классификатора
    if hasattr(model, 'random_state') or index in ['RF', 'ET', 'XGBoost', 'LightGBM']:
        # если у класса есть параметр random_state, задаем его (SEED должен быть доступен в функции)
        clf_tmp = model(random_state=SEED).fit(pred_train, veg_train)
    else:
        clf_tmp = model().fit(pred_train, veg_train)

    # создадим селектор для отбора наиболее информативных предикторов
    selector = SelectFromModel(estimator=clf_tmp, threshold='mean', prefit=True)

    # COMPROMISE: 'srtm', т.е., высота над уровнем моря, удаляется по VIF выше или 
    # по importance ниже, но без неё результаты прогнозирования оказываются
    # недостаточно точными из-за малого разрешения данных о погоде. 
    # сохраним данные о высотах над уровнем моря
    srtm_train = pred_train['srtm']
    srtm_valid = pred_valid['srtm']

    # Отильтруем признаки в тренировочном и валидационном наборах. 
    # Сначала создадим маску для столбцов, ...
    selected_features = pred_train.columns[selector.get_support()]
    # ...затем оставим в обучающем и валидационном наборах только столбцы... 
    # ...с самыми информативными признаками. 
    pred_train_current = pred_train[selected_features]
    pred_valid_current = pred_valid[selected_features]
    # COMPROMISE: 'srtm', т.е., высота над уровнем моря, удаляется по VIF выше или 
    # по importance ниже, но без неё результаты прогнозирования оказываются
    # недостаточно точными из-за малого разрешения данных о погоде. 
    # добавим вручную данные о высотах над уровнем моря
    if 'srtm' not in pred_train_current.columns:
        pred_train_current['srtm'] = srtm_train
        pred_valid_current['srtm'] = srtm_valid
    # создадим список имён оставшихся (самых информативных) столбцов
    retained_columns = pred_train_current.columns

    print(f"Было признаков: {default_featurs_num}, стало: {pred_train_current.shape[1]}")

    '''
    Определение функции 'objective' для последующей передачи в Optuna и последующее обучение модели. 
    По результатам обучения выводится отчёт о сравнении обученной модели с фиктивной и её гиперпараметрах. 
    '''
    def objective(trial): 
        params = parameters(trial)
        # Модель: передаём параметры, ...
        trained_model = model(**params)
        # ...обучаем на обучающих данных (*_train) и... 
        trained_model.fit(pred_train_current, veg_train)
        # составляем на основе обученной модели прогноз по валидационному набору предикторов.
        predicted = trained_model.predict(pred_valid_current) 

        # Расчёт целевой метрики на валидационном наборе данных ('veg_valid' и 'predicted_rf' [см. выше]). 
        if target_metric == 'roc_auc': 
            t_metric = roc_auc_score(veg_valid, predicted)
        elif target_metric == 'fbeta':
            t_metric = fbeta_score(veg_valid, predicted, beta=BETA)
        else: 
            print('Выберите правильную метрику; ожидается "roc_auc" или "fbeta". ')
        
        return t_metric      

    # создание обучалки
    study = optuna.create_study(direction="maximize", study_name=study_name)

    # Обучение. `with` необходимо для корректной работы, иначе код время от времени вылетает с ошибкой. 
    with np.errstate(under='ignore'):
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    # проверка результатов: вывод на дисплей гиперпараметров лучшей модели
    print(f'Лучшие параметры для {algorythm}: {study.best_params}.') 
    print()

    # Для сравнения выведем ROC-AUC фиктивной модели и... 
    print(f'Лучшее значение метрики {target_metric} для фиктивной модели {round(t_metric_dummy, 4)}.', sep='')
    print()
    # ...значение ROC-AUC для обученной.
    print(f'Лучшее значение метрики {target_metric} для {algorythm} {round(study.best_value, 4)}.', sep='')
    print()

    '''
    Создание модели (обучение на обучающем наборе данных), расчёт целевой метрики 
    и прочих (zccuracy, precision, recall). 
    '''
    best_model = model(**study.best_params).fit(pred_train_current, veg_train)

    # рассчитаем и запишем значения метрик
    valid_preds = best_model.predict(pred_valid_current)
    if target_metric == 'roc_auc': 
        best_metrics.loc[index, 'ROC-AUC'] = roc_auc_score(veg_valid, valid_preds)
    elif target_metric == 'fbeta': 
        best_metrics.loc[index, 'Fbeta'] = fbeta_score(veg_valid, valid_preds, beta=BETA)
    else: 
        print('Ошибка в выборе метрики или заголовке столбца. ')
    best_metrics.loc[index, 'accuracy'] = accuracy_score(veg_valid, best_model.predict(pred_valid_current))
    best_metrics.loc[index, 'precision'] = precision_score(veg_valid, best_model.predict(pred_valid_current))
    best_metrics.loc[index, 'recall'] = recall_score(veg_valid, best_model.predict(pred_valid_current))

    '''
    Сравнение обученной модели с лучшей на момент начала обучения. Если новая модель имеет 
    более высокую целевую метрику по сравнению с лучшей до того, то делаем лучшей её. 
    '''
    # сравнение с лучшей на данный момент моделью 
    # Если лучшая модель даёт меньшее значение ROC-AUC, чем для только что обученная, то...
    if t_metric_best < study.best_value: 
        # ...меняем прежнее лучшее значение ROC-AUC на ROC-AUC только что обученной модели, ...
        t_metric_best = study.best_value 
        # сохраняем в 'best_algorythm' 'random forest', 
        best_algorythm = algorythm
        # ...а саму лучшую модель меняем на случайный лес, добавив туда наилучшие гиперпараметры.
        model_best = model(**study.best_params)
        # для получения стабильных результатов вручную устанавливаем в модели `random_state=SEED`
        model_best.set_params(**{'random_state': SEED})
        # заменим имена столбцов на те, которые нужны для работы лучшей модели
        best_retained_columns = retained_columns

        # перемешиваем данные для model_best во избежание переобученности из-за строгого порядка строк
        X_tr_shf, y_tr_shf = shuffle(pred_train_current, veg_train, random_state=SEED)
        # обучаем модель
        model_best.fit(X_tr_shf, y_tr_shf)

    # # рассчитаем значения целевой метрики для подачи на дисплей
    # if target_metric == 'roc_auc':
    #     valid_preds = model_best.predict_proba(pred_valid[best_retained_columns])[:, 1]
    #     final_score = roc_auc_score(veg_valid, valid_preds)
    # else: 
    #     # если целевая метрика Fbeta, можно оставить 'predict'
    #     valid_preds = model_best.predict(pred_valid[best_retained_columns])
    #     final_score = fbeta_score(veg_valid, valid_preds, beta=BETA)
        
    # print(f'Лучшее значение метрики {target_metric} на валидационном наборе после обучения модели: {final_score:5.4G}')  
        
    return t_metric_best, model_best, best_algorythm, best_retained_columns