import itertools
import os
from typing import Literal

import alphashape
from html2image import Html2Image
import geopandas as gpd
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from shapely.geometry import Point

# загрузить конфиг
with open('C:/Users/user/Yandex.Disk/Важные документы/Исходные данные для статей/Моделирование вспышек/Моделирование ареала пихты/Abies_01/config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# загрузить адрес корневой директории
ROOT_DIR = config['ROOT_DIR']

def plot_predictions_results(
        # результаты прогнозирования функцией 'get_predictions_for_scenario'
        # для режима 'scenario' кортеж с прогнозами для трёх периодов
        # для других режимов кортеж с прогнозами для четырёх сценариев
        # HARDCODE: задавая формат данных, мы исходили из того, что функция будет обрабатывать 
        # результаты исследования четырёх траекторий SSP. Если это количество изменится, 
        # код придётся переписывать. 
        # NB: для корректной работы функции 'plot_predictions_results' функция 
        # 'get_predictions_for_scenario' должна была работать с аргументом obtain = mmm. 
        predictions: tuple[pd.Series, pd.Series, pd.Series] | list[tuple, tuple, tuple, tuple], 
        # координаты полигонов
        coordinates: pd.DataFrame, 
        # высоты н.у.м.
        altitudes: pd.DataFrame, 
        # строка с именем сценария (для сообщений и подписей)
        scenario_name: str | None, 
        # строка с исследованным периодом (для сообщений и подписей)
        period_limits: str, 
        # режим работы функции (определяет, какие карты будут выведены на график)
        # NB: для корректной работы функции при mode = scenario аргумент predictions 
        # должен быть кортежем из трёх pd.Series, в других случаях на вход должен быть 
        # подан список из четырёх кортежей. 
        mode: Literal['scenario', 'minimal', 'median', 'maximal'], 
        # выводить ли карту в ноутбук
        showmap: bool = False
): 
    # создадим объекты, нужные для работы при любом значении 'mode', кроме 'scenario'
    if mode != 'scenario': 
        # массив с обозначениями траекторий SSP
        labs = np.repeat(['3.4', '4.5', '7.0', '8.5'], len(coordinates))
        # координаты по долготе
        coord_x = list(itertools.chain.from_iterable([coordinates.iloc[:, 0]] * 4))
        # координаты по широте
        coord_y = list(itertools.chain.from_iterable([coordinates.iloc[:, 1]] * 4))
        # высоты над уровнем моря
        alts = list(itertools.chain.from_iterable([altitudes.values] * 4))
        # ширина графика
        width = 1000
    else: 
        # массив с обозначениями результатов симуляций (минимальная, медианная, максимальная)
        labs = np.repeat(['minimal', 'median', 'maximal'], len(coordinates))
        # координаты по долготе
        coord_x = list(itertools.chain.from_iterable([coordinates.iloc[:, 0]] * 3))
        # координаты по широте
        coord_y = list(itertools.chain.from_iterable([coordinates.iloc[:, 1]] * 3))
        # высоты над уровнем моря
        alts = list(itertools.chain.from_iterable([altitudes.values] * 3))
        # ширина графика
        width = 1500

    # если нужны наихудшие результаты симуляций по всем сценариям
    if mode == 'minimal': 
        # Выбираем прогнозы для наихудших (минимальное количество благоприятных пикселей) 
        # симуляций. В кортеже они идут первыми. 
        prediction_res = np.concatenate(
            [
                # конвертируем нули и единицы в True и False
                predictions[0][0].astype('bool').values, 
                predictions[1][0].astype('bool').values, 
                predictions[2][0].astype('bool').values, 
                predictions[3][0].astype('bool').values
            ]
        )
        # создаём заголовок карты
        header = f"Results for {period_limits}, most pessimistic simulation"
    # если нужны медианные результаты симуляций по всем сценариям
    elif mode == 'median':
        # Выбираем прогнозы для медианных (медианное количество благоприятных пикселей) 
        # симуляций. В кортеже они идут вторыми. 
        prediction_res = np.concatenate(
            [
                # конвертируем нули и единицы в True и False
                predictions[0][1].astype('bool').values, 
                predictions[1][1].astype('bool').values, 
                predictions[2][1].astype('bool').values, 
                predictions[3][1].astype('bool').values
            ]
        )
        # создаём заголовок карты
        header = f"Results for {period_limits}, median simiulation"
    # если нужны оптимальные результаты симуляций по всем сценариям
    elif mode == 'maximal': 
        # Выбираем прогнозы для наилучших (максимальное количество благоприятных пикселей) 
        # симуляций. В кортеже они идут третьими. 
        prediction_res = np.concatenate(
            [
                # конвертируем нули и единицы в True и False
                predictions[0][2].astype('bool').values, 
                predictions[1][2].astype('bool').values, 
                predictions[2][2].astype('bool').values, 
                predictions[3][2].astype('bool').values
            ]
        )
        # создаём заголовок карты
        header = f"Results for {period_limits}, most optimistic simulation"
    # если хотим показать минимум, медиану и оптимум для одной траектории SSP
    elif mode == 'scenario': 
        # Создаём список, где результаты минимальной, медианной и оптимальной симуляций 
        # идут друг за другом. Нули и единицы конвертируем в булевы переменные. 
        prediction_res = np.concatenate([series.astype('bool').values for series in predictions]).tolist()
        # создаём заголовок карты
        header = f"Prediction for {scenario_name}, time span {period_limits}"
    # если значение опции не разрешено, сообщаем об ошибке
    else: 
         raise ValueError('Неверно введён режим работы функции. \n')
        
    # создаём таблицу исходных данных для отрисовки
    sim_all = pd.DataFrame(
        {
            # координаты
            'point_x': coord_x, 
            'point_y': coord_y, 
            # результаты прогнозирования в булевых значениях
            'predictions': prediction_res, 
            # Метка симуляции: наихудшая, медианная, наилучшая для 'scenario' 
            # или коды траектории SSP для остальных опций в 'mode'. 
            'label': labs, 
            # высоты над уровнем моря
            'srtm': alts
            }
        ).reset_index(drop=True)

    # добавляем в таблицу данных для отрисовки данные о высотах
    # каждому столбцу соответствует свой диапазон высот
    sim_all['mountain_1'] = (sim_all['srtm'] >= 300) & (sim_all['srtm'] < 900)
    sim_all['mountain_2'] = (sim_all['srtm'] >= 900) & (sim_all['srtm'] < 1600)
    sim_all['mountain_3'] = (sim_all['srtm'] >= 1600) & (sim_all['srtm'] < 2000)
    sim_all['mountain_4'] = (sim_all['srtm'] >= 2000) & (sim_all['srtm'] < 2500)
    sim_all['mountain_5'] = sim_all['srtm'] > 2500

    # создаём объекты, нужные для оформления рисунка
    # пятиуровневая палитра для отрисовки изогипс на карте: от орехового к чёрному
    altitude_colors = {
        'mountain_1': '#B78453',    # Средне-коричневый (ореховый)
        'mountain_2': '#95673C',    # Насыщенный коричневый (каштановый)
        'mountain_3': '#724B25',    # Глубокий коричневый
        'mountain_4': '#42240C',    # Тёмно-коричневый (горький шоколад)
        'mountain_5': '#000000'     # Чёрный
    }
    # словарь для подписей уровней высот в легенде 
    altitude_labels = {
        'mountain_1': '300–600 m',
        'mountain_2': '600–900 m',
        'mountain_3': '900–1200 m',
        'mountain_4': '1200–1500 m',
        'mountain_5': 'higher 1500 m'
    }

    # создаём список 
    labels_list = list(sim_all["label"].unique())
    # определяем количество уникальных сценариев (подграфиков)
    num_plots = len(labels_list)

    # --- ШАГ 1: ДИНАМИЧЕСКОЕ ОПРЕДЕЛЕНИЕ ГЕОМЕТРИИ СЕТКИ И ПОДПИСЕЙ ---
    if mode == "scenario":
        # Для сценариев строим всё в ОДИН ряд (1 строка, n столбцов)
        num_rows = 1
        num_cols = num_plots
        # HARDCODE: подписи субграфиков
        custom_labels = {"minimal": "Minimal result", "median": "Median result", "maximal": "Maximal result"}
    else:
        # Во всех остальных случаях строим строго в ДВА столбца
        num_cols = 2
        num_rows = (num_plots + 1) // 2  # Округление вверх (для 4 элементов -> 2 строки, для 3 -> 2 строки)
        # HARDCODE: подписи субграфиков
        custom_labels = {"3.4": "SSP 3.4", "4.5": "SSP 4.5", "7.0": "SSP 7.0", "8.5": "SSP 8.5"}

    # формируем список заголовков для субграфиков
    subplot_titles = [custom_labels.get(str(lbl), str(lbl)) for lbl in labels_list]

    # Создаем пустую сетку с нужной геометрией
    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.06,
        vertical_spacing=0.1 if num_rows > 1 else 0.0,  # Отступ между строками нужен только при многорядной сетке
    )

    # Переменная для контроля отображения легенды предсказаний (True/False)
    show_pred_legend = True
    # True: условия для пихты благоприятны; False: модель предсказала неблагоприятные условия
    color_map = {True: "#228B22", False: "#FFFFF0"}

    # --- ШАГ 3: РАСЧЕТ И АВТОМАТИЧЕСКАЯ ОТРИСОВКА КОНТУРОВ ВЫСОТ ---
    # инициализируем трекер: False означает, что данная высота ЕЩЕ ВООБЩЕ НЕ ПОКАЗЫВАЛАСЬ в легенде
    legend_tracker = {alt: False for alt in altitude_colors}
    # создадим список ступеней высот для использования его в цикле отрисовки изогипс
    mountains = ["mountain_1", "mountain_2", "mountain_3", "mountain_4", "mountain_5"]

    # цикл отрисовки изогипс
    for altitude in mountains:
        # выбираем все точки, находящиеся в соответствующем диапазоне высот 'altitude'
        feature_points = sim_all[sim_all[altitude] == True]
        # на всякий случай: если диапазон высот пуст, переходим к следующей altitude
        if feature_points.empty:
            continue

        # вычисление альфа-формы (границ высотных зон)
        # Создаём набор точек (объектов shapely), 
        # которые находятся в диапазоне высот altitude с координатами x и y. 
        geometry = [Point(xy) for xy in zip(feature_points["point_x"], feature_points["point_y"])]
        # создадим фрейм Geopandas
        gdf_points = gpd.GeoDataFrame(feature_points, geometry=geometry)
        # создадим список кортежей с координатами точек
        points_coords = list(zip(gdf_points.geometry.x, gdf_points.geometry.y))

        # создаём границу вокруг точек заданного диапазона высот (изогипсу)
        # HARDCODE: alpha=15 подобран вручную так, чтобы развалить единый контур на мультиполигоны
        contour_poly = alphashape.alphashape(points_coords, alpha=15.0)
        # создаём список полигонов, полученных в предыдущей строке кода
        polygons_to_draw = list(contour_poly.geoms) if contour_poly.geom_type == "MultiPolygon" else [contour_poly]

        # отрисовываем контур во всех созданных ранее субграфиках
        # для каждого субграфика idx
        for idx in range(num_plots):
            # если mode='scenario' (3 субграфика в ряд)
            if mode == "scenario": 
                # определяем номер строки (первая и единственная)
                r = 1
                # определяем номер столбца
                c = idx + 1
            else: 
                # определяем номер строки целочисленным делением
                r = (idx // num_cols) + 1
                # определяем номер столбца через остаток от деления
                c = (idx % num_cols) + 1

            # для каждого полигона (изогипсы), который нужно отрисовать
            for poly in polygons_to_draw:
                # создаём список долгот точек изогипсы
                lon = list(poly.exterior.xy[0])
                # создаём список широт точек изогипсы
                lat = list(poly.exterior.xy[1])

                # проверяем: если для ЭТОЙ высоты запись в легенде еще не создана
                show_this_time = not legend_tracker[altitude]

                # отрисовка изогипсы
                # добавляем новый графический объект (с помощью add_trace)
                fig.add_trace(
                    # создаём объект
                    go.Scatter(
                        # долготы точек
                        x=lon,
                        # широты точек
                        y=lat,
                        # задаёт объект в виде линий (не отдельных точек)
                        mode="lines",
                        # задаёт внешний вид линий
                        line=dict(color=altitude_colors[altitude], width=1, dash="solid"),
                        # задаёт имя, под которым эта группа изогипс будет отображена в легенде
                        name=altitude_labels[altitude],
                        # показываем только если show_this_time == True
                        showlegend=show_this_time, 
                        legendgroup=altitude,
                    ), 
                    # Отрисовываем на субграфике из строки 'r'...
                    row=r,
                    # ...и столбца 'c'. 
                    col=c,
                )
                
                # Как только отрисовали самый первый элемент (полигон), 
                # жестко блокируем показ в легенде для всех последующих полигонов и субграфиков. 
                if show_this_time:
                    legend_tracker[altitude] = True

    # --- ШАГ 2: КАРТИРОВАНИЕ ТОЧЕК ПО СЕТКЕ ---
    # 'idx' – номер сценария (для определения места субграфика на рисунке)
    # 'label_val' – метка сценария
    for idx, label_val in enumerate(labels_list):
        # расчет индексов ячеек зависит от количества столбцов
        # если 'scenario', то один столбец
        if mode == "scenario":
            # определяем номер строки (первая и единственная)
            row = 1
            # определяем номер столбца
            col = idx + 1
        else:
            # определяем номер строки целочисленным делением
            row = (idx // num_cols) + 1
            # определяем номер столбца через остаток от деления
            col = (idx % num_cols) + 1

        # выбираем данные для отрисовки на текущем субграфике соответственно метке
        sub_data = sim_all[sim_all["label"] == label_val]

        # добавляем маркеры, обозначающие результат прогноза
        for pred_status in [True, False]:
            # Сначала выбираем те из них, которые соответствуют текущему значению 
            # 'pred_status' (положительный прогноз [True] или отрицательный [False]). 
            pred_data = sub_data[sub_data["predictions"] == pred_status]
            # На всякий случай – если данных с таким 'pred_status' нет, то переходим
            # на следующий шаг цикла. 
            if pred_data.empty:
                continue

            # То же самое, если 'pred_status' == False, т.е., прогноз негативный: 
            # такие точки просто не отрисовываем. 
            if pred_status == False: 
                continue

            # для положительных прогнозов добавляем точки
            fig.add_trace(
                # создаём объект plotly
                go.Scatter(
                    # долгота точки
                    x=pred_data["point_x"],
                    # широта точки
                    y=pred_data["point_y"], 
                    # способ отрисовки (точками)
                    mode="markers",
                    # внешний вид точек
                    marker=dict(color=color_map[pred_status], size=5), 
                    # имя в легенде
                    name="favorable for fir" if pred_status else "hostile for fir",
                    legendgroup=f"pred_{pred_status}",
                    # отключить показ в легенде цвета для неблагоприятных участков
                    showlegend=False if not pred_status else show_pred_legend
                ), 
                # номер строки, в которой находится субграфик
                row=row,
                # номер столбца, в котором находится субграфик
                col=col,
            )
        show_pred_legend = False

    # задаём высоту рисунка
    height=400 * num_rows

    # задаём отступ легенды от нижнего субграфика
    # HARDCODE: величина отступов подобрана вручную
    if mode == 'scenario': 
        y = -0.3
    else: 
        y = -0.1

    # задаём внешний вид графика
    fig.update_layout(
        # заголовок
        title=header,
        # высота графика
        height=height, 
        # ширина графика
        width=width, 
        # задаём ширину нижнего поля, резервируя место под легенду
        margin=dict(b=100 + (20 * num_rows)), 
        # позиционирование легенды строго по центру внизу
        legend=dict(
            # горизонтальное отображение элементов в линию
            orientation="h", 
            # привязка верха легенды к точке, заданной парой чисел (x, y)
            yanchor="top",
            # отступ под нижнюю ось
            y=y, 
            # центровка легенды по горизонтали
            xanchor="center",
            # выравнивание по центру всей ширины контейнера
            x=0.5, 
            # Задаём фиксированный размер маркеров в легенде, 
            # независимо от их размеров на самом графике. 
            itemsizing = 'constant'
        )
    )

    # Стилизуем шрифты подписей над графиками
    for anno in fig.layout.annotations:
        anno.font = dict(size=12, color="black", family="Arial")

    # настраиваем подписи легенды для участков с подходящими/неподходящими условиями для пихты
    for trace in fig.data:
        if trace.name == 'True':
            # подпись вместо True
            trace.name = 'favorable for fir' 
        elif trace.name == 'False':
            # подпись вместо False
            trace.name = 'hostile for fir'  

    # настраиваем внешний вид маркеров
    fig.update_traces(
        marker=dict(
            # небольшой размер
            size=2,      
            # лёгкая прозрачность
            opacity=0.35   
        )
    )
    
    # добавляем заголовок оси абсцисс (title) и задаём цвет сетки (gridcolor)
    fig.update_xaxes(title="Longitude (°E)", gridcolor='gray')
    fig.update_yaxes(gridcolor='gray')

    # убираем дублирующиеся подписи осей
    if mode == 'scenario':
        # код для отображения подписи оси ординат только у крайнего левого субграфика
        for axis in fig.layout:
            # отбор элементов графика, названия которого начинаются с 'yaxis'
            if axis.startswith('yaxis'):
                # если это крайний левый элемент
                if axis == 'yaxis':
                    fig.layout[axis].title.text = "Latitude (°N)"
                # если это следующие за ним элементы (yaxis2 etc.)
                else:
                    fig.layout[axis].title.text = None
    else: 
        for axis in fig.layout:
            # отбор элементов графика, названия которого начинаются с 'yaxis'
            if axis.startswith('yaxis'):
                # если это крайний левый элемент
                if axis == 'yaxis' or axis == 'yaxis3':
                    fig.layout[axis].title.text = "Latitude (°N)"
                # если это другие элементы (yaxis2 etc.)
                else:
                    fig.layout[axis].title.text = None
            # отбор элементов графика, названия которого начинаются с 'xaxis' 
            if axis.startswith('xaxis'): 
                # для нижних субграфиков (3 и 4) оставляем подписи осей абсцисс
                if axis == 'xaxis3' or axis == 'xaxis4': 
                    fig.layout[axis].title.text = 'Longitude (°E)'
                # для верхних (не xaxis3 или xaxis4) зануляем
                else:
                    fig.layout[axis].title.text = None

    # создадим имена промежуточных и итоговых файлов
    if mode != 'scenario': 
        file_name_html = mode + '_in_' + period_limits + '.html'
        file_name_png = mode + '_in_' + period_limits + '.png'
    else: 
        file_name_html = scenario_name + '_in_' + period_limits + '.html'
        file_name_png = scenario_name + '_in_' + period_limits + '.png'

    # сохраним рисунок в формате html
    fig.write_html(os.path.join(ROOT_DIR, 'future_predictions/maps', file_name_html), include_plotlyjs="cdn")

    # убираем все отступы у html-страницы, чтобы не было огромных полей
    custom_css = """
    body, html {
        margin: 0;
        padding: 0;
        overflow: hidden; /* убирает полосы прокрутки */
        background-color: transparent; /* делает фон прозрачным */
    }
    """

    # выставляем опцию, исключающую наложение элементов осей друг на друга
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)

    # создаём объект, конвертирующий html в графический формат
    hti = Html2Image()
    # настраиваем путь для сохранения
    hti.output_path = os.path.join(ROOT_DIR, 'future_predictions/maps')
    # делаем скриншот html-рисунка
    hti.screenshot(
        # путь к html
        html_file=os.path.join(ROOT_DIR, 'future_predictions/maps', file_name_html), 
        # имя сохраняемого файла
        save_as=file_name_png, 
        # зададим css для удаления полей
        css_str=custom_css, 
        # зададим размеры сохраняемого рисунка
        size=(width, height)
        )

    # если выбрана опция показа карты на экране
    if showmap:
        # отображаем итоговый совмещенный рисунок
        fig.show()
    else: 
        print('Карта сохранена на диск. ')
