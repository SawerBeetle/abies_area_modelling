def weather_means(data, feature, month):
    # выберем столбцы с заголовками, содержащими метку нужного признака
    feature_subset = data.filter(regex=feature)
    # Выберем из столбцов с описанием нужного признака те, 
    # которые содержат номер нужного месяца. 
    month_subset = feature_subset.filter(regex='_'+month+'_')
    # рассчитаем среднее значение признака для каждой строки
    mean_values = month_subset.astype('float').mean(axis=1, skipna=True, numeric_only=True)
    
    # вернём результат
    return(mean_values)