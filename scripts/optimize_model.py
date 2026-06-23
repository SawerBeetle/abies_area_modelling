import json
from lightgbm import LGBMClassifier
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, fbeta_score  
from xgboost import XGBClassifier

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
    def objective(trial):    
        params = parameters(trial)
        # Модель: передаём параметры, ...
        trained_model = model(**params)
        # ...обучаем на обучающих данных (*_train) и... 
        trained_model.fit(pred_train, veg_train)
        # составляем на основе обученной модели прогноз по валидационному набору предикторов.
        predicted = trained_model.predict(pred_valid) 

        # Расчёт целевой метрики на валидационном наборе данных ('veg_valid' и 'predicted_rf' [см. выше]). 
        # roc_auc = roc_auc_score(veg_valid, predicted)
        if target_metric == 'roc_auc': 
            t_metric = roc_auc_score(veg_valid, predicted)
        elif target_metric == 'fbeta':
            t_metric = fbeta_score(veg_valid, predicted, beta=BETA)
        else: 
            print('Выберите правильную метрику. ')
        
        return t_metric        
        # return roc_auc

    # создание обучалки
    study = optuna.create_study(direction="maximize", study_name=study_name)

    # Обучение. `with` необходимо для корректной работы, иначе код время от времени вылетает с ошибкой. 
    with np.errstate(under='ignore'):
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    # проверка результатов: вывод на дисплей гиперпараметров лучшей модели
    print(f'Лучшие параметры для {algorythm}: {study.best_params}.') 
    print()

    # Для сравнения выведем ROC-AUC фиктивной модели и... 
    # print('Лучшее значение ROC-AUC для фиктивной модели {:5.4G}'.format(roc_auc_dummy), '.', sep='')
    print(f'Лучшее значение метрики {target_metric} для фиктивной модели {round(t_metric_dummy, 4)}.', sep='')
    print()
    # ...значение ROC-AUC для модели случайного леса.
    # print(f'Лучшее значение ROC-AUC для {algorythm} {round(study.best_value, 4)}.', sep='')
    print(f'Лучшее значение метрики {target_metric} для {algorythm} {round(study.best_value, 4)}.', sep='')
    print()

    # создадим классификатор с лучшими параметрами
    best_model = model(**study.best_params).fit(pred_train, veg_train)

    # рассчитаем и запишем значения метрик
    if target_metric == 'roc_auc': 
        best_metrics.loc[index, 'ROC-AUC'] = study.best_value
    elif target_metric == 'fbeta': 
        best_metrics.loc[index, 'Fbeta'] = study.best_value
    else: 
        print('Ошибка в выборе метрики или заголовке столбца. ')
    best_metrics.loc[index, 'accuracy'] = accuracy_score(veg_valid, best_model.predict(pred_valid))
    best_metrics.loc[index, 'precision'] = precision_score(veg_valid, best_model.predict(pred_valid))
    best_metrics.loc[index, 'recall'] = recall_score(veg_valid, best_model.predict(pred_valid))

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

    # Выводим на экран значение рассчитанной для валидационного набора данных ROC-AUC 
    # для лучшей модели после всех замен (или их отсутствия) – для сравнения. 
    if target_metric == 'roc_auc':
        print('Лучшее значение метрики ', target_metric, ' в целом {:5.4G}'.\
            format(roc_auc_score(veg_valid, 
                                model_best.fit(pred_train, veg_train).\
                                predict(pred_valid))), 
            '.', sep='')
    else: 
        print('Лучшее значение метрики ', target_metric, ' в целом {:5.4G}'.\
            format(fbeta_score(veg_valid, 
                                model_best.fit(pred_train, veg_train).\
                                predict(pred_valid), beta=BETA)), 
            '.', sep='')        
        
    return t_metric_best, model_best, best_algorythm