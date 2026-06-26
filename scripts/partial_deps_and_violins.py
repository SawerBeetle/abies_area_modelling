import os

import json
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import math
from scipy.stats import ks_2samp
import seaborn as sns
from sklearn.inspection import PartialDependenceDisplay
from statannotations.Annotator import Annotator
from statannotations.stats.StatTest import StatTest

from format_fn import format_fn

# загрузить конфиг
with open('C:/Users/user/Yandex.Disk/Важные документы/Исходные данные для статей/Моделирование вспышек/Моделирование ареала пихты/Abies_01/config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# загрузить флаг сохранения рисунков в папку 'eda'
EDA = config['EDA']
# загрузить целевую метрику
TARGET_METRIC = config['TARGET_METRIC']
# загрузить адрес корневой директории
ROOT_DIR = config['ROOT_DIR']
# проверим, будет ли обучаться модель
TRAIN_MODEL = config['TRAIN_MODEL']

def partial_deps_and_violins(
        # модель, для которой определяем частные зависимости
        model, 
        # тестовый набор предикторов
        pred_test, 
        # тестовый набор зависимых переменных (данные о породном составе лесов)
        veg_test, 
        # путь к каталогу
        path_to_folder
        ): 
    
    # рассчитаем нужное количество столбцов
    if pred_test[model.feature_names_in_].shape[1] < 7: 
        n_cols = 3
        width = 10.5
    else: 
        n_cols = 4
        width = 14
    # рассчитаем нужное количество строк и высоту графика
    n_rows = math.ceil(pred_test.shape[1] / n_cols)
    height = 3 * n_rows

    # создаём график частных зависимостей
    part_dep = PartialDependenceDisplay.from_estimator(
        estimator=model, 
        X=pred_test, 
        features=list(pred_test.columns), 
        n_cols=n_cols, 
        n_jobs=-1
        )

    # создаём объект для форматирования подписей по оси абсцисс
    formatter = ticker.FuncFormatter(format_fn)
    # Для каждого графика...
    for ax in part_dep.axes_.flat: 
        if ax is not None: 
            # 1. Получаем текущую минимальную и максимальную точки шкалы Y
            ymin, ymax = ax.get_ylim()
            # 2. Вычисляем точный центр оси ординат
            y_center = ymin + (ymax - ymin) / 2            
            # ...рисуем горизонтальную линию на уровне 0.5, ...
            ax.axhline(y=y_center, color='r', linestyle='--', linewidth=.5)
            # ...форматируем подписи при засечках, ...
            ax.xaxis.set_major_formatter(formatter)
            # ...устанавливаем количество засечек не более 4, ...
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
            # ...задаём размеры засечек (чтобы не терялись на фоне rugs) по оси абсцисс и...
            ax.tick_params(axis='x', which='major', width=2, length=8)
            # ...задаём размер шрифта при засечках на обеих осях. 
            ax.tick_params(axis='both', labelsize=7)

    # выводим график на экран и сохраняем
    part_dep.figure_.set_size_inches(width, height)
    part_dep.figure_.subplots_adjust(hspace=0.35)

    # сохраняем график частных зависимостей в папку с модельью
    plt.savefig(
        os.path.join(path_to_folder + '/partial_dependencies.jpeg'), 
        bbox_inches='tight'
        )
    # выводим график частных зависимостей на экран
    plt.show()

    # создадим фрейм с данными для скрипичных графиков (будет передан в sns.violinplot)
    data_for_plot = pred_test.copy()
    # добавим туда данные о доминирующей лесообразующей породе
    data_for_plot['vegetation'] = veg_test
    # удалим строки с дублирующимися индексами
    # NOTE: чёрт его знает, зачем это нужно; возможно, стоит убрать
    data_for_plot = data_for_plot[~data_for_plot.index.duplicated(keep='first')]
    # создадим список предикторов, для которых будут построены скрипичные графики
    features = list(data_for_plot.columns[0:(data_for_plot.shape[1] - 1)])

    # создадим общий график (fig) и вложенные графики (ax)
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(width, height))
    # для удобства дальнейшей обработки сделаем массив вложенных графиков одномерным
    ax_flat = ax.flatten()

    # Создадим пользовательский тест 
    # для сравнения признаков пихтовых и прочих древостоев по Колмогорову – Смирнову. 
    custom_ks_test = StatTest(
        ks_2samp,
        test_long_name='Kolmogorov-Smirnov',
        test_short_name='K-S',
        stat_name='D'
    )

    # В цикле создадим скрипичные графики для всех features 
    # и разместим их во вложенных графиках axis. 
    for feature, axis in zip(features, ax_flat): 
        # установим метки оси абсцисс (по преобладающей породе)
        axis.set_xticklabels(['other', 'fir'])
        # создадим скрипичный график
        sns.violinplot(
            # данные для графика
            data=data_for_plot, 
            # группирующая переменная
            x='vegetation',
            # зависимая переменная (предиктор)
            y=feature, 
            # группирующий признак для выбора цветов
            hue='vegetation',
            # субграфик, на который попадёт построенный скрипичный график
            ax=axis, 
            # что будет отрисовано внутри "скрипки" (в данном случае, медиана и квартили)
            inner="quartile", 
            # обрезка данных по реальным максимуму и минимуму
            cut=0, 
            # цвета для окрашивания "скрипок"
            palette = ['green', 'purple'], 
            # не показывать легенду
            legend=None
            )
        # Создадим аннотации к скрипичным графикам 
        # с уровнем значимости различий по Колмогорову-Смирнову. 
        # список кортежей, указывающих сравниваемые пары (здесь она одна, и кортеж один)
        pairs = [(0, 1)]
        # создание аннотатора
        annot = Annotator(
            # субграфик, к которому подготовлена аннотация
            axis, 
            # сравниваемые пары (т.е., единственная пара)
            pairs, 
            # фрейм с данными
            data=data_for_plot, 
            # группирующая переменная
            x='vegetation', 
            # сравниваемая переменная
            y=feature
            )
        # создание аннотации; 'verbose=0' нужно для предотвращения вывода текста в консоль
        annot.configure(test=custom_ks_test, text_format='star', loc='outside', verbose=0)
        # передача аннотации на субграфик
        annot.apply_and_annotate()
    
    # предотвратим наложение субграфиков
    plt.tight_layout()
    # сохраним скрипичные графики в папку с моделью
    plt.savefig(os.path.join(path_to_folder + '/violins.jpeg'))
    # выведем скрипичные графики на экран
    plt.show()
