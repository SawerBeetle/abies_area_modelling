# Prediction of Abies sibirica area shift under changed climate of the Middle Siberia

**This project have been created with Russian Science Foundation and Krasnoyarsk Regional Fund of Science and Technology Support, project No. 25-14-20068 funding.** 

*Code*: D.A. Demidko. 

*Data providing*: A.S. Shuspanov. 

*Supervision*: V.I. Kharuk. 

## Basic idea of the project

The project have been developed to check what parts of region between 49.5 to 60.0°N, and 80.0 to 100.0°E will be suitable for conifer tree species *Abies sibirica* Ledeb. (Siberian fir) under future climate changes. We assume the relief and climate features are responsible for the suitability. Initially, we have trained some models where domination of Siberian fir is modelled by relief and climate predictors. Next, we pick the best model and use it to predict Siberian fir-dominatid forests area under different climate changes scenarios. 

## Data structure

The root directory contains the follow files and directories: 

* **config.json**: the global parameters for data processing and models training (e.g., target metric or path); 
* **poetry.lock**: Python package dependencies; 
* **prediction.ipynb**: the Jupiter notebook with a code to predict future changes of *Abies sibirica* environment; 
* **pyproject.toml**: project configuration; 
* **README.md**: You read it; 
* **train_models**: the Jupiter notebook with a code for data processing and models training;
* **data**: raw and and processed data about forest vegetation, climate and relief; 
* **eda**: the results of exploratory data analysis and some plots describing data; 
* **models**: the directories with trained models of the *Abies sibirica* spatial distribution and some supplementary files; 
* **models_comparison**: files describing the comparative traits of the models; 
* **scripts**: the Python scripts for the different stages of data processing. 

### **data** content 

Inside the directory contains the files 

* **abies.csv** and **not_abies.csv**: the sets (10000 each) for fir-dominated and other tree species-dominated polygons of processed climate and relief data, incl. polygon ID and coordinates, altitude, surface curvature, slope steepness and orientation and monthly averages for soil water content, precipitation, snow depth, temperature, evaporation and relative humidity. 
* **predictors_abies.txt** and **predictors_not_abies.txt**: the whole sets of raw climate and relief data for *Abies* and other species-dominated poligons; 
* **predictors_abies.xml** and **predictors_not_abies.xml**: the metadata for above-mentioned whole sets of data. 

### **eda** content

The directory contains two `ydata-profiling` generated EDA reports: 

* **comparison.html**: the comparative statistical descriptions of fir-dominated and other species-dominated polygons and correlation structures for these two groups; 
* **non_multicollinear_pred_analysis.html**: the same for predictors after high-VIF elimination. 

The figures inside the directory are: 

* **feature_clusters.jpeg**: the results of cluster analysis for relief and climate predictors (about naming rules see: `train_models.ipynb`, Section 2.1); 
* **violins.jpeg**: violin plots to compare fir-dominated and other species-dominated polygons; asterisks denote the statistical significance of Kolmogorov – Smirnov tests. 

### **models** content

Each directory inside **models** contains the best model optimized with some target metric (*ROC-AUC* or *Fbeta*, where *beta* is within [0.6, 0.7, 0.8, 0.9]). The files in these directories are: 

* **abies_area_model_*datetime*_*target_metric***: best model; 
* **data_for_map.csv**: the selected predictors for 20000 forested polygons (a number of fir- and other tree species-dominated polygons is equal), geographical coordinates of the polygons (`point_x` and `point_y`), real (`vegetation_real`) and predicted (`vegetation_predicted`) data about dominate tree species, where `1` indicates *A*. *sibirica* domination, and results of prediction by the model (`prognosis_res`). 
* **log_abies_area_model_*datetime*.txt**: the log file with some data about the model training procedure and model performance; 
* **map.jpeg**: the map of forest vegetation for 20000 forested polygons used for training; 
* **partial_dependencies.jpeg**: partial dependencies plot for the selected predictors. 

### **