import os
import time

import geopandas as gpd
import json
import numpy as np
import pandas as pd
import plotly.express as px

# загрузить конфиг
with open('C:/Users/user/Yandex.Disk/Важные документы/Исходные данные для статей/Моделирование вспышек/Моделирование ареала пихты/Abies_01/config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# загрузить коэффициент бета
MODEL_NAME = config['MODEL_NAME']
# загрузить путь к корневой папке
ROOT_DIR = config['ROOT_DIR']

def create_html_map(
        lon: np.ndarray | pd.Series, 
        lat: np.ndarray | pd.Series, 
        classes: np.ndarray | pd.Series, 
        color_map: dict[str, str], 
        title_initial: str, 
        path_to_target: str
        ): 
    # cоздаем основу для регулярную сетки точек в градусах
    if not isinstance(lon, np.ndarray): 
        lon = np.array(lon)
    if not isinstance(lat, np.ndarray):
        lat = np.array(lat)
    # создаём массив классов, отображаемых на карте
    if not isinstance(classes, np.ndarray):
        classes = np.array(classes)

    # инициализируем исходный GeoDataFrame
    gdf_points = gpd.GeoDataFrame(
        # классы
        {'class_id': classes},
        # координаты в градусах
        geometry=gpd.points_from_xy(lon, lat),
        # проекция
        crs="EPSG:4166"
    )

    # заменяем точечную геометрию на полигональную
    gdf_cells = gdf_points.copy()
    # HACK: статичное значение буфера – это плохо, но расчёт для каждой точки 
    # приводит к тому, что объединить их в полигоны не получается. 
    gdf_cells['geometry'] = gdf_points.geometry.buffer(0.01)

    # объединяем смежные полигоны с одинаковым class_id
    gdf_polygons = gdf_cells.dissolve(by='class_id', as_index=False)

    # разделяем несвязанные полигоны (MultiPolygon -> Polygon)
    gdf_polygons = gdf_polygons.explode(index_parts=False).reset_index(drop=True)

    # упрощаем геометрию полигонов для ускорения рендеринга
    gdf_polygons['geometry'] = gdf_polygons['geometry'].simplify(tolerance=0.01, preserve_topology=True)
    # добавляем ID полигонов
    gdf_polygons['id'] = gdf_polygons.index.astype(str)

    # выводим отчёт о начале работы
    print("Шаг 1: Конвертируем полигоны GeoPandas в формат GeoJSON...")
    start_time = time.time()

    # извлекаем интерфейс геометрии отдельно
    geojson_data = gdf_polygons.__geo_interface__

    # выводим отчёт о продолжении работы
    print(f"Конвертация завершена за {time.time() - start_time:.2f} сек.")
    print("Шаг 2: Передаем данные в Plotly и генерируем карту...")

    # Вычисляем центральную точку для каждого полигона сетки
    centroids = gdf_polygons.geometry.centroid

    # Записываем долготу и широту центроидов в датафрейм (их размер гарантированно совпадет)
    gdf_polygons['lon'] = centroids.x
    gdf_polygons['lat'] = centroids.y

    # строим карту
    fig = px.choropleth(
        # данные
        gdf_polygons, 
        # геоданные
        geojson=geojson_data, 
        # метки цвета (= результат прогнозирования)
        color='class_id', 
        # метки полигонов
        locations='id', 
        # цветовая карта
        color_discrete_map=color_map, 
        # картографическая проекция
        projection="azimuthal equal area", 
        # вид карты
        scope='world',
        # заголовок
        title=f"{title_initial}; model name is {MODEL_NAME}",
        # информация, содержащаяся во всплывающей подсказке
        hover_data={'lon': True, 'lat': True, 'class_id': True, 'id': True}
        )
    
    # отображение региона исследования
    fig.update_geos(
        # принудительно включаем отображение только нужного региона
        fitbounds="locations"
    )
    fig.update_traces(
    # удаление белых линий на стыках полигонов (там, где между ними возникают зазоры)
        marker_line_width=0, 
        hovertemplate=(
            "<b>ID региона:</b> %{location}<br>"
            "<b>Результат (Class ID):</b> %{customdata[2]}<br>"  # Индекс зависит от порядка в hover_data
            "<span style='color:gray;'>---------------------------</span><br>"
            "<b>Широта (Lat):</b> %{customdata[1]:.4f}°<br>"    # Округление до 4 знаков
            "<b>Долгота (Lon):</b> %{customdata[0]:.4f}°<br>"   # Округление до 4 знаков
            "<extra></extra>"  # Этот тег скрывает стандартную боковую плашку Plotly
        )
    )

    # отчёт о начале рендеринга
    print("Карта успешно построена! Рендеринг в браузере...")
    # вывод карты на дисплей
    fig.show()
    # сохранение карты на диск
    fig.write_html(os.path.join(ROOT_DIR, path_to_target))
