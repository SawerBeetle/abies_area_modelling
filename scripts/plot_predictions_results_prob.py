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

def plot_predictions_results_prob(
        # результаты прогнозирования функцией 'get_predictions_for_scenario'
        # для режима 'scenario' кортеж с прогнозами для трёх периодов
        # для других режимов кортеж с прогнозами для четырёх сценариев
        # predictions: list[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], 
        predictions: list[pd.DataFrame, pd.DataFrame, pd.DataFrame], 
        # координаты полигонов
        coordinates: pd.DataFrame, 
        # высоты н.у.м.
        altitudes: pd.DataFrame, 
        # данные о современном расположении пихтовых древостоев
        fir_points: pd.DataFrame, 
        # строка с исследуемым периодом в формате 'YYYY–YYYY' (для сообщений и подписей)
        period_limits: str, 
        # количество симуляций, по которым строились прогнозы
        n_simulations: int = 100, 
        # строковые данные с именами траекторий SSP (для сообщений и подписей)
        # HACK: здесь изменена размерность списка и имена траекторий SSP, 
        # т.к. по ходу одну из них (3.4) решили не обрабатывать вообще. 
        # TODO: возможно, потом будет добавлена ещё модель (модели), и придётся вернуться 
        # к динамической обработке данных. 
        # scenarios: list[str, str, str, str] = ['SSP 3.4', 'SSP 4.5', 'SSP 7.0', 'SSP 8.5'], 
        scenarios: list[str, str, str] = ['SSP 4.5', 'SSP 7.0', 'SSP 8.5'], 
        # выводить ли карту в ноутбук
        showmap: bool = False,
        # Изменять ли прозрачность точек с результатами прогноза 
        # искусственно (т.е., задействовать ли scale_array) относительно значений по умолчанию, 
        # где точки с нулевой вероятностью абсолютно прозрачны, 
        # со 100%-й абсолютно непрозрачны (нужно для более наглядного представления результатов). 
        scale_opacity = False
): 

    # Определим функцию для масштабирования значений вероятностей того, что участки 
    # будут благоприятны для пихты (чтобы на карте они выглядели более заметными). 
    def scale_array(arr):
        # Создаем копию массива, чтобы не менять исходный, и приводим к float
        scaled = np.array(arr, dtype=float)
        
        # Находим маску для всех элементов, которые строго больше нуля
        pos_mask = scaled > 0

        # если положительных элементов нет, возвращаем как есть
        if not np.any(pos_mask):
            return scaled  
            
        # выделяем только положительные элементы
        pos_vals = scaled[pos_mask]

        # находим значения минимального и максимального элементов
        min_val = pos_vals.min()
        max_val = pos_vals.max()
        
        # защита от деления на ноль, если все ненулевые элементы одинаковы
        if max_val == min_val:
            scaled[pos_mask] = 1.0
        else:
            # линейно масштабируем диапазон [min_val, max_val] в диапазон [0.1, 1.0]
            scaled[pos_mask] = 0.05 + 0.95 * (pos_vals - min_val) / (max_val - min_val)
            
        return scaled

    # 1. ---Создаём объекты и переменные, нужные для построения графика---

    # объединим все данные о прогнозах в один фрейм
    predictions_df = pd.concat(predictions, ignore_index=True)
    # рассчитаем количество положительных прогнозов для каждого полигона (равно сумме по строке)
    prediction_res = np.sum(predictions_df, axis=1)
    # создадим метки сценариев
    labs = np.repeat(scenarios, len(coordinates))
    # размножим вектор координат так, чтобы они имелись для каждого полигона при каждом сценарии
    coord_x = list(itertools.chain.from_iterable([coordinates.iloc[:, 0]] * len(scenarios)))
    coord_y = list(itertools.chain.from_iterable([coordinates.iloc[:, 1]] * len(scenarios)))
    # размножим вектор высот н.у.м. так, чтобы они имелись для каждого полигона при каждом сценарии
    alts = list(itertools.chain.from_iterable([altitudes.values] * len(scenarios)))
    # создадим заголовок рисунка
    header = f"Prediction for {period_limits}"
    # зададим ширину рисунка
    width = 800

    # находим экстремумы исходных координат, чтобы жестко зафиксировать границы субграфиков
    lon_min = float(coordinates.iloc[:, 0].min())
    lon_max = float(coordinates.iloc[:, 0].max())
    lat_min = float(coordinates.iloc[:, 1].min())
    lat_max = float(coordinates.iloc[:, 1].max())
    
    # добавим небольшой отступ (5%), чтобы крайние точки не прижимались к рамке субграфиков
    lon_buffer = (lon_max - lon_min) * 0.05
    lat_buffer = (lat_max - lat_min) * 0.05
    
    # рассчитаем размеры субграфиков для их нормальной отрисовки
    map_lon_range = [lon_min - lon_buffer, lon_max + lon_buffer]
    map_lat_range = [lat_min - lat_buffer, lat_max + lat_buffer]
    
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

    # на всякий случай, конвертируем значения координат в формат с плавающей точкой
    # TODO: при возможности проверить, нужен ли этот код вообще
    sim_all['point_x'] = pd.to_numeric(sim_all['point_x'], errors='coerce').astype(float)
    sim_all['point_y'] = pd.to_numeric(sim_all['point_y'], errors='coerce').astype(float)

    # добавляем данные о сгруппированных высотах
    sim_all['mountain_1'] = (sim_all['srtm'] >= 300) & (sim_all['srtm'] < 900)
    sim_all['mountain_2'] = (sim_all['srtm'] >= 900) & (sim_all['srtm'] < 1600)
    sim_all['mountain_3'] = (sim_all['srtm'] >= 1600) & (sim_all['srtm'] < 2000)
    sim_all['mountain_4'] = (sim_all['srtm'] >= 2000) & (sim_all['srtm'] < 2500)
    sim_all['mountain_5'] = sim_all['srtm'] > 2500

    # добавляем палитру для отрисовки изогипс
    altitude_colors = {
        'mountain_1': '#B78453',    # Средне-коричневый (ореховый)
        'mountain_2': '#95673C',    # Насыщенный коричневый (каштановый)
        'mountain_3': '#724B25',    # Глубокий коричневый
        'mountain_4': '#42240C',    # Тёмно-коричневый (горький шоколад)
        'mountain_5': '#000000'     # Чёрный
    }

    # словарь для подписей изогипс в легенде 
    altitude_labels = {
        'mountain_1': '300–600 m',
        'mountain_2': '600–900 m',
        'mountain_3': '900–1200 m',
        'mountain_4': '1200–1500 m',
        'mountain_5': 'higher 1500 m'
    }

    # определяем количество уникальных сценариев (подграфиков)
    num_plots = len(scenarios)

    # определим количество строк и столбцов, в которых будут размещены субграфики
    # HARDCODE: по ходу работы я решил, что три графика лучше разместить в сетке 
    # 2×2, оставив пустую ячейку под легенду. 
    # TODO: потом, возможно, стоит вернуться 
    # к динамическому расчёту количества строк и столбцов. 
    # количество столбцов
    num_cols = 2
    # количество строк
    # округление вверх (для 4 элементов -> 2 строки, для 3 -> 2 строки)
    num_rows = (num_plots + 1) // 2 

    # зададим высоту рисунка
    # NB: расчёт высоты должен быть именно здесь, т.к. 'num_rows' определяется в предыдущей строке кода
    height = 400 * num_rows


    # Создаём список из трёх заголовков субграфиков (имена траекторий SSP), 
    # а четвёртый заголовок оставляем пустым; на месте этого субграфика будет легенда. 
    # HARDCODE: этот код предполагает, что графики для трёх климатических траекторий 
    # будут расположены в сетке 2×2. 
    # TODO: при возможном возвращении к динамической обработке размерности графика 
    # этот код станет бесполезным. 
    titles_2x2 = scenarios + [""]

    # создаём полотно, разделённое на субграфики
    fig = make_subplots(
        # количество строк
        rows=num_rows,
        # количество столбцов
        cols=num_cols,
        # заголовки субграфиков
        subplot_titles=titles_2x2, 
        # горизонтальный и вертикальный промежутки между субграфиками
        horizontal_spacing=0.08,
        vertical_spacing=0.15
    )

    # 2. ---Создаём мультиполигон ареала пихты, не нанося его пока на карту---

    # создаем пустой объект для будущего полигона
    fir_poly_gdf = gpd.GeoDataFrame() 
    # Выполняем, если фрейм с координатами полигонов, где сейчас доминирует пихта, существует, 
    # и если он не пуст. 
    if fir_points is not None and not fir_points.empty:
        # очищаем временную копию, чтобы не портить исходный объект в памяти Jupyter
        fir_cleaned = fir_points.dropna(subset=["point_x", "point_y"])

        # если фрейм с координатами после удаления NA не пустой
        if not fir_cleaned.empty:
            # создаём объект с точечными геометриями
            geometry_fir = [Point(xy) for xy in zip(fir_cleaned["point_x"], fir_cleaned["point_y"])]
            # из этого объекта создаём фрейм GeoPandas, чтобы далее можно было создавать полигоны
            gdf_points_fir = gpd.GeoDataFrame(fir_cleaned, geometry=geometry_fir)
            # для alphashape конвертируем фрейм в список
            points_coords_fir = list(zip(gdf_points_fir.geometry.x, gdf_points_fir.geometry.y))

            # строим альфа-форму
            # HARDCODE: значение alpha=15 определено эмпирически и позволяет 
            # наиболее правдоподобно делить контур на мультиполигоны. 
            contour_poly_fir = alphashape.alphashape(points_coords_fir, alpha=15.0)

            # Создаём и сохраняем в отдельную переменную мультиполигон fir_poly_gdf 
            # вместо перезаписи fir_points. 
            fir_poly_gdf = gpd.GeoDataFrame(geometry=[contour_poly_fir], crs="EPSG:4326")

    # 3. ---Создаём и отрисовываем изогипсы---

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
            # определяем номер строки целочисленным делением
            r = (idx // num_cols) + 1
            # определяем номер столбца через остаток от деления
            c = (idx % num_cols) + 1

            # для каждого полигона (изогипсы), который нужно отрисовать
            for poly in polygons_to_draw:
                # создаём список долгот точек изогипсы
                lon = np.array(poly.exterior.xy[0], dtype=float).tolist()
                # создаём список широт точек изогипсы
                lat = np.array(poly.exterior.xy[1], dtype=float).tolist()

                # проверяем: если для ЭТОЙ высоты запись в легенде еще не создана
                show_this_time = not legend_tracker[altitude]

                # отрисовка изогипсы
                # добавляем новый графический объект (с помощью add_trace)
                fig.add_trace(
                    # создаём объект
                    go.Scatter(
                        # координаты
                        x=lon,
                        y=lat, 
                        # отрисовка линий
                        mode="lines",
                        # внешний вид линий
                        line=dict(color=altitude_colors[altitude], width=1, dash="solid"), 
                        # прозрачность
                        opacity=0.75, 
                        # имя для линии каждого вида (в легенду)
                        name=altitude_labels[altitude],
                        # показывать ли легенду (показываем только если show_this_time == True)
                        showlegend=show_this_time,  
                        legendgroup=altitude,
                    ), 
                    # в каком столбце/в какой строке графика отображаем
                    row=r,
                    col=c,
                )
                
                # Как только отрисовали самый первый элемент (полигон), 
                # блокируем показ в легенде для всех последующих полигонов и субграфиков. 
                if show_this_time:
                    legend_tracker[altitude] = True

    # 4. ---Наносим на карту результаты прогнозирования---

    # 'idx' – номер сценария (для определения места субграфика на рисунке)
    # 'label_val' – метка сценария
    for idx, label_val in enumerate(scenarios):
        # определяем номер строки целочисленным делением
        row = (idx // num_cols) + 1
        # определяем номер столбца через остаток от деления
        col = (idx % num_cols) + 1

        # создаём выборку данных для данного сценария 'label_val'
        sub_data = sim_all[sim_all["label"] == label_val]

        # если прозрачность точек с положительными прогнозами не масштабируется
        if not scale_opacity:
            # Задаём зелёный цвет, прозрачность по умолчанию 
            # (чем выше вероятность, тем ниже прозрачность). Деление на 'n_simulations' нужно для 
            # приведения вероятности (0-'n_simulations') к масштабу, который требует значение opacity (0-1). 
            rgba_colors = [f"rgba(0, 128, 0, {val})" for val in (sub_data['predictions'] / n_simulations)]
        # если решили масштабировать прозрачность точек
        else:
            # Рассчитаем для каждого объекта (пикселя) вероятность того, что в будущем 
            # условия там будут благоприятны для пихты. 
            prob_values = scale_array(sub_data['predictions'] / n_simulations)
            # Зададим цвет пикселей, наносимых на карту, с учётом определённых в предыдущей строке 
            # вероятностей, с помощью которых зададим прозрачность. 
            rgba_colors = [f"rgba(0, 128, 0, {val})" for val in prob_values]

            # FIXME: этот код закомментирован, т.к. дублирует код создания полигонов для 
            # текущего распространения пихтовых лесов, находяящийся выше (на момент написания 
            # комментария стр. 206-210). Если при исполнении функции она упадёт – раскомментировать. 
            # TODO: разобраться, нужен ли этот код вообще. 
            # geometry_fir = [Point(xy) for xy in zip(fir_points["point_x"], fir_points["point_y"])]
            # gdf_points_fir = gpd.GeoDataFrame(fir_points, geometry=geometry_fir)
            # points_coords_fir = list(zip(gdf_points_fir.geometry.x, gdf_points_fir.geometry.y))
            
        # 4.1. ---Расчёт процента покрытия современного ареала прогнозом сценария 'label_val'---
        # исходное значение перекрытия принимается равным нулю
        overlap_percent = 0.0

        # работаем, если все нужные объекты существуют и не пустые
        if fir_points is not None and not fir_poly_gdf.empty and not sub_data.empty:
            # Превращаем координаты точек с в GeoDataFrame: ... 
            # ...сначала превращаем их в точечные объекты GeoPandas, ... 
            geometry_sim = [Point(xy) for xy in zip(sub_data["point_x"], sub_data["point_y"])]
            # ...а потом делаем из них фрейм с геоданными. 
            gdf_all_points = gpd.GeoDataFrame(sub_data, geometry=geometry_sim, crs="EPSG:4326")

            # фильтруем точки: оставляем только те, что внутри фиолетового контура современного ареала
            points_inside_fir = gpd.sjoin(gdf_all_points, fir_poly_gdf, how="inner", predicate="within")

            # если объект с точками внутри современного ареала пихты не пустой, работаем дальше
            if not points_inside_fir.empty:
                # получаем количество точек внутри современного ареала
                total_points_in_fir = len(points_inside_fir)
                # считаем точки с ненулевым прогнозом
                favorable_count = len(points_inside_fir[points_inside_fir["predictions"] > 0])
                # сычисляем итоговый процент
                overlap_percent = (favorable_count / total_points_in_fir) * 100
        # 4.1 ---конец---

        # 4.2 ---Добавим на субграфик данные о результатах симуляций---
        fig.add_trace(
            # дополняем рисунок, используя точечный объект Plotly
            go.Scatter(
                # долготы
                x=sub_data["point_x"],
                # широты
                y=sub_data["point_y"],                
                # способ нанесения данных (точки)
                mode="markers",
                # Внешний вид точек (здесь используем ранее созданный объект 'rgba_colors'
                # с информацией о цвете и прозрачности). 
                marker=dict(color=rgba_colors, size=10),
                # настроим всплывающую подсказку
                hovertemplate=(
                    "Longitude (X): %{x:.4f}<br>"
                    "Latitude (Y): %{y:.4f}<br>"
                    "Altitude: %{customdata[1]} m<br>"
                    "Favorable predictions: %{customdata[0]}%<br>"
                    #  <extra></extra> скрывает дефолтное название трейса сбоку
                    "<extra></extra>" 
                ), 
                # данные о результате прогноза и высоте н.у.м. 
                customdata=pd.DataFrame({'prob': sub_data['predictions'], 'alt': sub_data['srtm']})
            ),
            # указываем, в какие строку и столбец общего графика помещаем информацию
            row=row,
            col=col,
        )
        # 4.2 ---конец---

        # 4.3 ---Добавим данные о современном положении древостоев с преобладанием пихты---
        # проверяем, не пуст ли созданный полигон 'fir_poly_gdf'
        if not fir_poly_gdf.empty: 
            # для каждой строки с номером '_' и содержимым 'row_geo'
            for _, row_geo in fir_poly_gdf.iterrows():
                # извлечём из строки полигон
                geom = row_geo.geometry
                # если он отсутствует в 'row_geo' или пуст, переходим на следующий шаг цикла
                if geom is None or geom.is_empty:
                    continue
                
                # превращаем мультиполигон в список полигонов
                polygons = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
                
                # Для каждого полигона, извлечённого в список 'polygons'...
                for poly in polygons:
                    # проверяем, достаточно ли в нём вершин для отрисовки; если их меньше трёх, ... 
                    if poly.exterior is None or len(poly.exterior.coords) < 3:
                        # ...переходим на следующий шаг цикла
                        continue
                    
                    # берем числовые массивы [0] и [1] и приводим к float
                    # TODO: разобраться, нужно ли это для работы функции; эта конвертация была 
                    # добавлена при одной из попыток исправить ошибку, и неочевидно, что 
                    # она чему-то помогла. 
                    x_coords = np.array(poly.exterior.xy[0], dtype=float).tolist()
                    y_coords = np.array(poly.exterior.xy[1], dtype=float).tolist()
                    
                    # добавляем "пихтовые" полигоны
                    fig.add_trace(
                        # используя точечный объект Plotly
                        go.Scatter(
                            # долготы
                            x=x_coords,
                            # широты
                            y=y_coords,
                            # способ отрисовки (линии, т.е., ограниченные ими полигоны)
                            mode="lines",         
                            # указываем, что полигон нужно залить цветом
                            fill="toself", 
                            # определяем цвет заливки
                            fillcolor="rgba(128, 0, 128, 0.6)", 
                            # определяем цвет и толщину границ полигона
                            line=dict(color="purple", width=2),
                            # имя слоя для легенды
                            name="Current fir domination", 
                            # легенду прямо сейчас не показываем (позже, в общей легенде)
                            showlegend=False, 
                            legendgroup="add_points", 
                            # добавляем к заливке паттерн из точек
                            fillpattern=dict(
                                # тип паттерна (собственно, точка)
                                shape=".", 
                                # плотность распределения точек
                                solidity=0.2, 
                                # размер точек
                                size=3, 
                                # цвет точек
                                fgcolor="purple", 
                                # задаём прозрачный фон под паттерном
                                bgcolor="rgba(0,0,0,0)" 
                            ), 
                            # запрещаем всплывающую подсказку
                            hoverinfo='skip'
                        ),
                        # номер строки, в которой находится обрабатываемый субграфик
                        row=row,
                        # то же, номер столбца
                        col=col
                    )
        # 4.3 ---конец---

        # 4.4 ---Добавляем легенду с информацией о перекрывании текущего и прогнозного ареалов пихты---
        # определяем индексы осей Plotly для текущего субграфика
        axis_suffix = f"{idx + 1}" if idx > 0 else ""
        
        # добавляем легенду
        fig.add_annotation(
            # текст
            text=f"Potentially favorable proportion of current range: {overlap_percent:.1f}%", 
            # Задаём привязку по горизонтали (domain указывает, что нужно привязываться к границам рамки, 
            # а не к числам на оси). 
            xref=f"x{axis_suffix} domain",  
            # задаём привязку по вертикали
            yref=f"y{axis_suffix} domain",  
            # сдвиг по горизонтали (3% от левого края графика)
            x=0.03, 
            # сдвиг по вертикали (15% от нижнего края)
            y=0.15, 
            # определяет, какой границей легенда должна прикасаться к точке привязки по горизонтали
            xanchor="left",
            # то же, по вертикали
            yanchor="top",
            # отключаем стрелку-указатель
            showarrow=False, 
            # определяем внешний вид шрифта
            font=dict(
                size=11,
                color="purple", 
                family="Arial",
                weight="bold" 
            ), 
            # белая полупрозрачная подложка, чтобы текст не сливался с картой
            bgcolor="rgba(255, 255, 255, 0.8)", 
            # цвет рамки
            bordercolor="purple",
            # толщина рамки
            borderwidth=1,
            # отступ внутрь от рамки
            borderpad=4
        )
        # 4.4 ---конец---

    # 5. ---Форматируем внешний вид общего графика, включая добавление легенды---
    fig.update_layout(
        # заголовок
        title=header,
        # высота
        height=height,  
        # ширина
        width=width, 
        # размер полей
        margin=dict(b=120, l=80, r=40, t=100), 
        
        # размещаем легенду строго внутри четвертой ячейки (строка 2, столбец 2)
        legend=dict(
            # вертикальное размещение элементов легенды
            orientation="v", 
            # центр легенды по вертикали размещён в якорной точке  
            yanchor="middle", 
            # центрируем якорную точку по вертикали в правой нижней четверти
            y=0.25, 
            # центр легенды по горизонтали размещён в якорной точке
            xanchor="center",       
            # центрируем легенду по горизонтали в правом столбце
            x=0.75, 
            # размер символов в легенде независим от размера на рисунке
            itemsizing='constant',
            # шрифт легенды
            font=dict(size=11, family="Arial")
        ), 
        # цвет подложки рисунка в целом
        paper_bgcolor='white',  
        # цвет подложки графиков
        plot_bgcolor='white',   
    )

    # жестко фиксируем географические границы для ВСЕХ субграфиков (осей 1, 2, 3 и 4)
    fig.update_xaxes(
        # цвет линий сетки
        gridcolor='gray',
        # главное: отключает автоматический расчёт границ
        autorange=False,       
        # границы графика будут выставлены согласно 'map_lon_range' (см. -1.-)
        range=map_lon_range    
    )
    fig.update_yaxes(
        # цвет линий сетки
        gridcolor='gray',
        # главное: отключает автоматический расчёт границ
        autorange=False,       
        # границы графика будут выставлены согласно 'map_lon_range' (см. -1.-)
        range=map_lat_range    
    )

    # настройка подписей осей для сетки 2х2 (Latitude слева, Longitude снизу)
    # HARDCODE: будет работать только для этой размерности (3 графика в сетке 2 на 2)
    # TODO: при необходимости заменить на динамический расчёт
    # строка 1, столбец 1 (график 1)
    fig.update_yaxes(title_text="Latitude (°N)", row=1, col=1)
    fig.update_xaxes(title_text=None, row=1, col=1)
    # строка 1, столбец 2 (график 2)
    fig.update_yaxes(title_text=None, row=1, col=2)
    fig.update_xaxes(title_text='Longitude (°E)', row=1, col=2) # Под вторым графиком пусто, ему нужна ось X
    # строка 2, столбец 1 (график 3)
    fig.update_yaxes(title_text="Latitude (°N)", row=2, col=1)
    fig.update_xaxes(title_text='Longitude (°E)', row=2, col=1)

    # зададим внешний вид маркеров в легенде
    for trace in fig.data:
        # 'trace.mode' проверяется по 'go.Scatter'
        if trace.mode == "markers":
            # отключаем отображение точечных слоёв в легенде
            trace.showlegend = False
            # задаём размер точки
            # TODO: проверить, не противоречит ли это свойствам маркера в блоке 4.2
            trace.marker.size = 2  

    # принудительно обновляем свойства маркеров
    fig.update_traces(
        marker=dict(
            # размер
            # TODO: проверить, не противоречит ли это свойствам маркера в блоке 4.2
            size=2, 
            # прозрачность
            # opacity=0.35   
        ),
        selector=dict(mode="markers")
    )

    # стилизация кастомных заголовков сценариев и фиолетовых плашек
    for anno in fig.layout.annotations:
        # проверяем, относится ли текст к заголовкам сценариев
        if anno.text in scenarios:
            anno.font = dict(size=13, color="black", family="Arial", weight="bold")
        # если это фиолетовая плашка с процентами, её шрифт мы НЕ трогаем (остается кастомным)
        elif "Potentially favorable" in str(anno.text):
            pass

    # создаём имена html-файла и графического файла
    file_name_html = 'probabilites_in_' + period_limits + '.html'
    file_name_png = 'probabilites_in_' + period_limits + '.png'

    # сохраним рисунок в формате html
    fig.write_html(os.path.join(ROOT_DIR, 'future_predictions/maps', file_name_html), include_plotlyjs="cdn")

    # убираем все отступы у страницы с помощью правильного CSS
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
        size=(width, height)
        )

    if showmap:
        # отображаем итоговый график
        fig.show()
    else: 
        print('Карта сохранена на диск. ')
