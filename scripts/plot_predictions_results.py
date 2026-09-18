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
from shapely.geometry import Point

# загрузить конфиг
with open('C:/Users/user/Yandex.Disk/Важные документы/Исходные данные для статей/Моделирование вспышек/Моделирование ареала пихты/Abies_01/config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# загрузить адрес корневой директории
ROOT_DIR = config['ROOT_DIR']

def plot_predictions_results(
        predictions: tuple[pd.Series, pd.Series, pd.Series] | list[tuple, tuple, tuple, tuple], 
        coordinates: pd.DataFrame, 
        altitudes: pd.DataFrame, 
        scenario_name: str | None, 
        period_limits: str, 
        mode: Literal['scenario', 'minimal', 'median', 'maximal'], 
        showmap: bool = False
): 

    if mode == 'minimal': 
        prediction_res = np.concatenate(
            [
                predictions[0][0].astype('bool').values, 
                predictions[1][0].astype('bool').values, 
                predictions[2][0].astype('bool').values, 
                predictions[3][0].astype('bool').values
            ]
        )
        labs = np.repeat(['3.4', '4.5', '7.0', '8.5'], len(coordinates))
        coord_x = list(itertools.chain.from_iterable([coordinates.iloc[:, 0]] * 4))
        coord_y = list(itertools.chain.from_iterable([coordinates.iloc[:, 1]] * 4))
        alts = list(itertools.chain.from_iterable([altitudes.values] * 4))
        header = f"Results for {period_limits}, most pessimistic simulation"
        width = 1900
    elif mode == 'median':
        prediction_res = np.concatenate(
            [
                predictions[0][1].astype('bool').values, 
                predictions[1][1].astype('bool').values, 
                predictions[2][1].astype('bool').values, 
                predictions[3][1].astype('bool').values
            ]
        )
        labs = np.repeat(['3.4', '4.5', '7.0', '8.5'], len(coordinates))
        coord_x = list(itertools.chain.from_iterable([coordinates.iloc[:, 0]] * 4))
        coord_y = list(itertools.chain.from_iterable([coordinates.iloc[:, 1]] * 4))
        alts = list(itertools.chain.from_iterable([altitudes.values] * 4))
        header = f"Results for {period_limits}, median simiulation"
        width = 1900
    elif mode == 'maximal': 
        prediction_res = np.concatenate(
            [
                predictions[0][2].astype('bool').values, 
                predictions[1][2].astype('bool').values, 
                predictions[2][2].astype('bool').values, 
                predictions[3][2].astype('bool').values
            ]
        )
        labs = np.repeat(['3.4', '4.5', '7.0', '8.5'], len(coordinates))
        coord_x = list(itertools.chain.from_iterable([coordinates.iloc[:, 0]] * 4))
        coord_y = list(itertools.chain.from_iterable([coordinates.iloc[:, 1]] * 4))
        alts = list(itertools.chain.from_iterable([altitudes.values] * 4))
        header = f"Results for {period_limits}, most optimistic simulation"
        width = 1900
    elif mode == 'scenario': 
        prediction_res = np.concatenate([series.astype('bool').values for series in predictions]).tolist()
        labs = np.repeat(['minimal', 'median', 'maximal'], len(coordinates))
        coord_x = list(itertools.chain.from_iterable([coordinates.iloc[:, 0]] * 3))
        coord_y = list(itertools.chain.from_iterable([coordinates.iloc[:, 1]] * 3))
        alts = list(itertools.chain.from_iterable([altitudes.values] * 3))
        header = f"Результаты для {scenario_name}, период {period_limits}"
        width = 1500
    else: 
         raise ValueError('Неверно введён режим работы функции. \n')
        
    #####
    # создаём таблицу исходных данных для отрисовки
    sim_all = pd.DataFrame(
        {
            # координаты
            'point_x': coord_x, 
            'point_y': coord_y, 
            # результаты прогнозирования в булевых значениях
            'predictions': prediction_res, 
            # метка симуляции (наихудшая, медианная, наилучшая)
            'label': labs, 
            # высоты над уровнем моря
            'srtm': alts
            }
        ).reset_index(drop=True)

    # добавляем данные о высотах
    sim_all['mountain_1'] = (sim_all['srtm'] >= 300) & (sim_all['srtm'] < 900)
    sim_all['mountain_2'] = (sim_all['srtm'] >= 900) & (sim_all['srtm'] < 1600)
    sim_all['mountain_3'] = (sim_all['srtm'] >= 1600) & (sim_all['srtm'] < 2000)
    sim_all['mountain_4'] = (sim_all['srtm'] >= 2000) & (sim_all['srtm'] < 2500)
    sim_all['mountain_5'] = sim_all['srtm'] > 2500

    # добавляем данные о высотах
    sim_all['mountain_1'] = (sim_all['srtm'] >= 300) & (sim_all['srtm'] < 900)
    sim_all['mountain_2'] = (sim_all['srtm'] >= 900) & (sim_all['srtm'] < 1600)
    sim_all['mountain_3'] = (sim_all['srtm'] >= 1600) & (sim_all['srtm'] < 2000)
    sim_all['mountain_4'] = (sim_all['srtm'] >= 2000) & (sim_all['srtm'] < 2500)
    sim_all['mountain_5'] = sim_all['srtm'] > 2500

    #####

    #####
    # создаём объекты, нужные для оформления рисунка

    # пятиуровневая палитра: от орехового к чёрному
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
        'mountain_5': 'Выше 1500 m'
    }
    #####

    #####
    # создаём рисунок

    # --- ШАГ 1: Создаем БАЗОВЫЙ рисунок ОДИН раз вне цикла ---
    fig = px.scatter(
        sim_all, 
        x="point_x",              
        y="point_y",              
        color="predictions",        
        color_discrete_map={True: '#228B22', False: '#FFFFF0'}, 
        facet_col="label",          
        title=header,
        height=400, 
        width=width
    )

    # --- НАСТРОЙКА ПОДПИСЕЙ ОТДЕЛЬНЫХ РИСУНКОВ ---
    # Словарь для красивого перевода технических названий из колонки "label"
    if mode == 'scenario':
        custom_labels = {
            'minimal': 'Minimal result',
            'median': 'Median result',
            'maximal': 'Maximal result'
        }
    else: 
        custom_labels = {
            '3.4': 'SSP 3.4', 
            '4.5': 'SSP 4.5', 
            '7.0': 'SSP 7.0', 
            '8.5': 'SSP 8.5'
        }

    # настраиваем подписи над субграфиками
    for anno in fig.layout.annotations:
        # Ищем старый текст, например "label=minimal"
        for tech_name, clear_name in custom_labels.items():
            if tech_name in anno.text:
                anno.text = clear_name  # Меняем текст подписи
                anno.font = dict(size=12, color="black", family="Arial") # Опционально: настраиваем шрифт

    # Флаги для легенды (по одному на каждую высоту)
    legend_tracker = {alt: False for alt in altitude_colors}

    # --- ШАГ 2: Запускаем цикл только для генерации и добавления контуров ---
    for altitude in ['mountain_1', 'mountain_2', 'mountain_3', 'mountain_4', 'mountain_5']:
        
        # 1. Отбираем точки для текущей высоты
        feature_points = sim_all[sim_all[altitude] == True]
        if feature_points.empty:
            continue # Пропускаем, если точек для этой высоты нет

        # 2. Переводим в геометрию
        geometry = [Point(xy) for xy in zip(feature_points['point_x'], feature_points['point_y'])]
        gdf_points = gpd.GeoDataFrame(feature_points, geometry=geometry)

        # 3. Извлекаем координаты и строим альфа-форму
        points_coords = list(zip(gdf_points.geometry.x, gdf_points.geometry.y))
        contour_poly = alphashape.alphashape(points_coords, alpha=15.0)

        # 4. Разбиваем мультиполигон, если необходимо
        polygons_to_draw = []
        if contour_poly.geom_type == 'MultiPolygon':
            polygons_to_draw = list(contour_poly.geoms)
        else:
            polygons_to_draw = [contour_poly]

        # 5. Добавляем контуры текущей высоты на созданный ранее fig во все 3 колонки
        if mode == 'scenario':
            for col_idx in [1, 2, 3]:
                for poly in polygons_to_draw:
                    lon = list(poly.exterior.xy[0])
                    lat = list(poly.exterior.xy[1])
                    
                    fig.add_trace(
                        go.Scatter(
                            x=lon,
                            y=lat,
                            mode="lines",
                            # Используем уникальный цвет для каждого уровня высоты
                            line=dict(color=altitude_colors[altitude], width=1, dash="solid"), 
                            name=altitude_labels[altitude],
                            # Показывать в легенде только один раз для этой высоты
                            showlegend=not legend_tracker[altitude], 
                            legendgroup=altitude       
                        ),
                        row=1, col=col_idx 
                    )
                    legend_tracker[altitude] = True # Выключаем дублирование в легенде
        else: 
            for col_idx in [1, 2, 3, 4]:
                for poly in polygons_to_draw:
                    lon = list(poly.exterior.xy[0])
                    lat = list(poly.exterior.xy[1])
                    
                    fig.add_trace(
                        go.Scatter(
                            x=lon,
                            y=lat,
                            mode="lines",
                            # Используем уникальный цвет для каждого уровня высоты
                            line=dict(color=altitude_colors[altitude], width=1, dash="solid"), 
                            name=altitude_labels[altitude],
                            # Показывать в легенде только один раз для этой высоты
                            showlegend=not legend_tracker[altitude], 
                            legendgroup=altitude       
                        ),
                        row=1, col=col_idx 
                    )
                    legend_tracker[altitude] = True # Выключаем дублирование в легенде            

    # --- ШАГ 3: Финальная настройка стилей (выполняется ОДИН раз в самом конце) ---
    fig.update_layout(
        # заголовок легенды
        legend_title_text="Marks: ", 
        legend=dict(
            # гарантирует правильный порядок слоев в легенде от 1 до 5
            traceorder="normal", 
            # обеспечивает хорошо видимый размер знака в легенде
            itemsizing="constant"
            ) 
    )

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

    # код для отображения подписи оси ординат только у крайнего левого субграфика
    for axis in fig.layout:
        # отбор элементов графика, названия которого начинаются с 'yaxis'
        if axis.startswith('yaxis'):
            # если это крайний левый элемент
            if axis == 'yaxis':
                fig.layout[axis].title.text = "Latitude (°N)"
            # если это следующие за ним элементы (yaxis2 etc.)
            else:
                fig.layout[axis].title.text = "" 

    # создадим имена промежуточных и итоговых файлов
    if mode != 'scenario': 
        file_name_html = mode + '_in_' + period_limits + '.html'
        file_name_png = mode + '_in_' + period_limits + '.png'
    else: 
        file_name_html = scenario_name + '_in_' + period_limits + '.html'
        file_name_png = scenario_name + '_in_' + period_limits + '.png'

    # сохраним рисунок в формате html
    fig.write_html(os.path.join(ROOT_DIR, 'future_predictions/maps', file_name_html), include_plotlyjs="cdn")

    # Убираем все отступы у страницы
    custom_css = """
    body, html {
        margin: 0;
        padding: 0;
        overflow: hidden; /* убирает полосы прокрутки */
        background-color: transparent; /* делает фон прозрачным, если нужно */
    }
    """

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
        size=(width, 400)
        )

    if showmap:
        # Отображаем итоговый совмещенный рисунок
        fig.show()
    else: 
        print('Карта сохранена на диск. ')
