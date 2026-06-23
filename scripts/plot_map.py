import matplotlib.ticker as ticker

def plot_map(
        # Холст (график), на котором будет отрисована карта; 
        # передаётся из plot_map_and_save_data. 
        ax, 
        # фрейм с данными для создания карты
        df, 
        # название признака (имя столбца), по которому будут окрашены точки
        feature, 
        # заголовок
        title, 
        # список цветов точек
        color_list, 
        # имена в легенде
        legend_names=None
        ): 
    # создадим словарь с соответствием значений признака из 'feature' и цветов из 'color_list'
    our_cmap = dict(zip(df[feature].unique(), color_list))
    # для каждой пары значений 'res' и цветов 'col' нанесём точки на холст 'ax'
    for res, col in our_cmap.items():
        # выберем все объекты с нужным значением
        subset = df[df[feature] == res]
        # нанесём точки с координатами 'point_x' и 'point_y'
        ax.scatter(subset['point_x'], subset['point_y'], c=col, marker='.', s=5, label=res)
        # отформатируем метки при осях
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0f}'))
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0f}'))
        # нанесём сетку
        ax.grid(True, alpha=0.4)
        # добавим заголовок
        ax.set_title(title)
    # Добавим в легенду имена значений: ...
    try:
        # ...назначенные явно, либо...
        ax.legend(labels=legend_names)
    except:
        # ...идущие по умолчанию. 
        ax.legend(labels=df[feature].unique())
        
    return ax