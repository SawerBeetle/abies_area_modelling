# Prediction of Abies sibirica area shift under changed climate of the Middle Siberia

**This project have been created with Russian Science Foundation and Krasnoyarsk Regional Fund of Science and Technology Support, project No. 25-14-20068 funding.** 

*Code*: D.A. Demidko. 

*Data providing*: A.S. Shuspanov. 

*Supervision*: V.I. Kharuk. 

## Basic idea of the project

The project have been developed to check what parts of region between 49.5 to 60.0°N, and 80.0 to 100.0°E will be suitable for conifer tree species *Abies sibirica* Ledeb. (Siberian fir) under future climate changes. We assume the relief and climate features are responsible for the suitability. Initially, we have trained some models where domination of Siberian fir is modelled by relief and climate predictors. Next, we pick the best model and use it to predict Siberian fir-dominatid forests area under different climate changes scenarios. 

## Project structure

The root directory contains the follow files and directories: 

* **gitattributes** and **gitignore**: git service files; 
* **config.json**: the global parameters for data processing and models training (e.g., target metric or path); 
* **pipeline_current_area.drawio**: XML-file with data processing and model training description by block diagram (for https://app.diagrams.net/); 
* **pipeline_current_area.drawio.png**: the block diagram with data processing and model training description; 
* **poetry.lock**: Python package dependencies; 
* **prediction.ipynb**: the Jupiter notebook with a code to predict future changes of *Abies sibirica* environment; 
* **pyproject.toml**: project configuration; 
* **README.md**: You read it; 
* **train_models**: the Jupiter notebook with a code for data processing and models training;
* **data**: raw and and processed data about forest vegetation, climate and relief; 
* **eda**: the results of exploratory data analysis and cluster plot describing relationships of the predictors; 
* **models**: the directories with trained models of the *Abies sibirica* spatial distribution and some supplementary files; 
* **models_comparison**: files describing the comparative traits of the models; 
* **scripts**: the Python scripts for the different stages of data processing. 
.
|____config.json
|____data
| |____abies.csv
| |____not_abies.csv
| |____predictors_abies.7z
| |____predictors_abies.txt
| |____predictors_abies.xml
| |____predictors_not_abies.7z
| |____predictors_not_abies.txt
| |____predictors_not_abies.xml
|____eda
| |____comparison.html
| |____feature_clusters.jpeg
| |____not_multicollinear_pred_analysis.html
|____map_current
| |____abies_range_interactive_map.html
| |____data.csv
| |____data.parquet
| |____metrics.txt
|____models
| |____abies_area_model_fbeta_0.6
| | |____abies_area_model_2026_07_01_14_30_fbeta
| | |____data_for_map.csv
| | |____log_abies_area_model_2026_07_01_14_30.txt
| | |____map.jpeg
| | |____partial_dependencies.jpeg
| | |____violins.jpeg
| |____abies_area_model_fbeta_0.7
| | |____abies_area_model_2026_07_01_14_33_fbeta
| | |____data_for_map.csv
| | |____log_abies_area_model_2026_07_01_14_33.txt
| | |____map.jpeg
| | |____partial_dependencies.jpeg
| | |____violins.jpeg
| |____abies_area_model_fbeta_0.8
| | |____abies_area_model_2026_07_01_14_36_fbeta
| | |____data_for_map.csv
| | |____log_abies_area_model_2026_07_01_14_36.txt
| | |____map.jpeg
| | |____partial_dependencies.jpeg
| | |____violins.jpeg
| |____abies_area_model_fbeta_0.9
| | |____abies_area_model_2026_07_01_14_39_fbeta
| | |____data_for_map.csv
| | |____log_abies_area_model_2026_07_01_14_39.txt
| | |____map.jpeg
| | |____partial_dependencies.jpeg
| | |____violins.jpeg
| |____abies_area_model_roc_auc
| | |____abies_area_model_2026_07_01_14_24_roc_auc
| | |____data_for_map.csv
| | |____log_abies_area_model_2026_07_01_14_24.txt
| | |____map.jpeg
| | |____partial_dependencies.jpeg
| | |____violins.jpeg
|____models_comparison
| |____comparisons.txt
| |____model_metrics.csv
|____pipeline_current_area.drawio
|____pipeline_current_area.drawio.png
|____poetry.lock
|____prediction.ipynb
|____pyproject.toml
|____README.md
|____scripts
| |____calculate_weather_data.py
| |____format_fn.py
| |____load_raw_data.py
| |____load_weather_completely.py
| |____load_weather_partially.py
| |____optimize_model.py
| |____partial_deps_and_violins.py
| |____plot_map.py
| |____plot_map_and_save_data.py
| |____prepare_datasets.py
| |____read_log.py
| |____remove_multicollinear.py
| |____save_best_model.py
| |____weather_means.py
|____train_models.ipynb

### **data** content 

Inside the directory contains the files 

* **abies.csv** and **not_abies.csv**: the sets for fir-dominated and other tree species-dominated polygons of processed climate and relief data, incl. polygon ID and coordinates, altitude, surface curvature, slope steepness and orientation and monthly averages for soil water content, precipitation, snow depth, temperature, evaporation and relative humidity. 
* **predictors_abies.txt** and **predictors_not_abies.txt**: the whole sets of raw climate and relief data for *Abies* and other species-dominated poligons; in github both these files are replaced by archived **predictors_abies.7z** and **predictors_not_abies.7z**. 
* **predictors_abies.xml** and **predictors_not_abies.xml**: the metadata for above-mentioned whole sets of data. 

### **eda** content

The directory contains two `ydata-profiling` generated EDA reports: 

* **comparison.html**: the comparative statistical descriptions of fir-dominated and other species-dominated polygons and correlation structures for these two groups; 
* **non_multicollinear_pred_analysis.html**: the same for predictors after high-VIF elimination. 

The figure inside the directory is: 

* **feature_clusters.jpeg**: the results of cluster analysis for relief and climate predictors (about naming rules see: `train_models.ipynb`, Section 2.1). 

### **map_current** content

The directory contains four files: 

* **abies_range_interactive_map.html**: the map visualizes the results of best model of *Abies sibirica*-dominated forests distribution (about best model see **metrics.txt** and **config.json**); 
* **data.csv** and **data.parquet**: the same data to create the **abies_range_interactive_map.html**; current version of data contains 
    * *pointid* (identificators of polygons),
    * *pre_01*, *pre_05* (January and May precipitation), 
    * *snow_depth_10* (October snow depth),
    * *evap_03*, *evap_04* (March and April evaporation), 
    * *rel_hum_09* (September relative air humidity),
    * *soil_water_03* (March soil water content in upper layer),
    * *srtm* (altitude),
    * *real_veg* (data about vegetation according to [Vega PRO](http://pro-vega.ru/) analysis of remote sensing data; *1* is fir-dominated forests, *0* is the forests dominated by other tree species),
    * *predicted_veg* (data about vegetation according to the best model prediction; *1* is fir-dominated forests, *0* is the forests dominated by other tree species),
    * *prognosis_res* (real vs. predicted vegetation data; *TP* is true positive prediction, *TN* – true negative prediction, *FP* and *FN* are false positive and false negative respectively),
    * *point_x* (longitude),
    * *point_y* (latitude). 
* **metrics.txt**: the file with the metrics (accuracy, precision and recall) of the best model on the full dataset. 

### **models** content

Each directory inside **models** contains the best model optimized with some target metric (*ROC-AUC* or *Fbeta*, where *beta* is within [0.6, 0.7, 0.8, 0.9]). The files in these directories are: 

* **abies_area_model_*datetime*_*target_metric***: best model; 
* **data_for_map.csv**: the selected predictors for 20000 forested polygons (a number of fir- and other tree species-dominated polygons is equal), geographical coordinates of the polygons (`point_x` and `point_y`), real (`vegetation_real`) and predicted (`vegetation_predicted`) data about dominate tree species, where `1` indicates *A*. *sibirica* domination, and results of prediction by the model (`prognosis_res`). 
* **log_abies_area_model_*datetime*.txt**: the log file with some data about the model training procedure and model performance; 
* **map.jpeg**: the map of forest vegetation for 20000 forested polygons used for training; 
* **partial_dependencies.jpeg**: partial dependencies plot for the selected predictors; 
* **violins.jpeg**: violin plots to compare fir-dominated and other species-dominated polygons; asterisks denote the statistical significance of Kolmogorov – Smirnov tests. 

### **models_comparison** content

The directory contains two files: 

* **comparisons.txt**: the comparison of prediction results of the models to each other; these models were compared by chi-square method and Kramer's V; 
* **model_metrics.csv**: the accuracy, precision and recall of each model; the models are sorted by rank sum of these metrics in ascending orderю 

### **scripts** content

The directory contains all the scripts used in **train_models.ipynb** and **prediction.ipynb**. For detail description see Section 1.2 in **train_models.ipynb**. 

## Data source and description

Источник: Ссылка на датасет (Kaggle, S3-баккет, база данных). Важно: никогда не заливайте огромные сырые файлы данных напрямую в Git (используйте DVC или .gitignore).Описание признаков: Краткий дата-дикт (перечень ключевых колонок, целевая переменная).Объем данных: Количество строк, столбцов, временной интервал.

## Pipeline and modelling

Предобработка (Preprocessing): Как обрабатывались пропуски, кодировались категориальные признаки, масштабировались числовые переменные.Модели (Models): Какие алгоритмы тестировались (например, Baseline — LogisticRegression, финальная модель — LightGBM).Метрики (Metrics): Какие метрики оптимизировались (ROC-AUC, RMSE, F1-score) и почему выбраны именно они (связь с бизнес-метриками).Результаты (Results): Таблица или график с финальным качеством моделей на валидационной/тестовой выборке.

## Installation and usage

Шаг 1: Клонирование и окружение

git clone https://github.com
cd project-name
python -m venv venv
source venv/bin/activate  # Для Windows: venv\Scripts\activate
pip install -r requirements.txt

Шаг 2: Загрузка / подготовка данных

Шаг 3: Обучение модели

Шаг 4: Инференс (Проверка работы)

## Tech stack

Список ключевых библиотек и технологий в виде бейджей или лаконичного списка:

