import gc

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# INFO: процедуру взятия среза (и агрументы 'start_yr' и 'end_yr' пришлось ввести в код, 
# потому что иначе не получалось обрабатывать большие массивы данных (а они были). 
# Пришлось экономить память таким образом. Это же касается команды 'del' и использоватия 'gc.collect'.
def plot_model_results(
        # обрабатываемый набор данных
        datasets: list[pd.DataFrame], 
        # заголовки элементов легенды
        labels: list[str], 
        # цвета линий
        colors: list[str], 
        # субграфик для размещения данных
        ax=None, 
        # год начала обрабатываемого временного интервала
        start_yr=2015, 
        # год конца обрабатываемого временного интервала
        end_yr=2099, 
        # заголовок (заголовки) субграфика
        title='2015–2099'
        ):

    # создадим словарь для сохранения результатов прогноза
    predictions = dict.fromkeys(labels)

    # в цикле обрабатываем набор данных, подставляем заголовки графиков, выбираем цвета
    for dataset, plot_label, col in zip(datasets, labels, colors): 

        # сначала делаем срез, чтобы обработать данные только для интересующего временного отрезка
        sub_df = dataset.loc[:, start_yr:end_yr].astype('float16')
        # INFO: Рассчитываем среднее значение по всем симуляциям, т.е. группируем данные 
        # по заголовкам столбцов. Каждый заголовок – год, всего для каждого года 100 симуляций. 
        # Далее усредняем эти 100 симуляций, получая среднее значение пригодности объекта (полигона) 
        # для обитания пихты. Полученное значение будет находиться между 0 и 1. 
        # Следующее усреднение даёт среднее значение уже для года по всем полигонам. 
        # Это и будет долей пригодных для пихты объектов-полигонов. 
        group_mean = sub_df.T.groupby(level=0).mean().T.mean()
        # заполним словарь для данного ключа (сценария CMIP)
        # predictions[plot_label] = sub_df.T.groupby(level=0).mean().T
        predictions[plot_label] = sub_df.T.groupby(level=0).mean().mean()
        # для построения графика регрессии создаём фрейм со средними по годам и собственно с годами
        to_regression = pd.concat(
            [group_mean.reset_index(drop=True), pd.Series(group_mean.index).reset_index(drop=True)], 
            axis=1, 
            ignore_index=True
        )

        # строим график изменения доли пригодных полигонов по годам
        sns.lineplot(data=group_mean, ax=ax, lw=.5, color=col)
        # строим график регрессии
        sns.regplot(
            x=to_regression.iloc[:, 1], 
            y=to_regression.iloc[:, 0], 
            ax=ax, 
            color=ax.lines[-1].get_color(), 
            label=plot_label,
            scatter=False, 
            ci=None
            )

        # INFO: эти процедуры нужны для очистки памяти, иначе график может не построиться
        del dataset, sub_df, to_regression, group_mean
        gc.collect()

        # создаём заголовок графика
        ax.set_title(title)
        # ставим подпись оси абсцисс
        ax.set_xlabel('Year')
        # запрещаем добавление легенды
        if ax.get_legend() is not None: 
            ax.get_legend().remove()

    return predictions