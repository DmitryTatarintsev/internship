![](https://github.com/DmitryTatarintsev/internship/blob/main/AlfaBankCampus/2023-10-15_23-44-57.png)

Тестовое задание: <a href='https://www.kaggle.com/competitions/alfabank-campus/overview'> alfabank-campus/overview </a> </br>

# Регрессионная модель временного ряда. Прогностическая модель AlfaBank Campus.
<a href='https://drive.google.com/file/d/1mg57JFFxk4CxFKScDmcQjw9J24lFkryz/view?usp=sharing'> Прогностическая модель. Data Science </a> </br>
<a href='https://drive.google.com/file/d/1RStwa5GgUjuH7uZhEHeUgskp-2z4VNQT/view?usp=sharing'> Прогностическая модель. Deep Learning </a> </br>
<a href='https://drive.google.com/file/d/1jnWmYqSfGZHiNiAfVDwyPQNH8Z_wEbr4/view?usp=sharing'> Итог. Submission </a> </br>

<a href='https://drive.google.com/file/d/1-8YpGrM3K62n4lYwnFLH9BVuGw4c9lr-/view?usp=sharing'> requirements. </a> </br>

Стэк: numpy, pandas, matplotlib, seaborn, tensorflow, keras, sklearn, xboost, catboost.

Статус: **Завершен.**

Цель: Нужно разработать регрессионную модель прогнозирования временного ряда в рамках соревнования AlfaBank Campus, которая предсказывает будущие траты клиента, используя информацию о совершенных тратах. Предсказать 10 значений.

Набор данных: <a href='https://www.kaggle.com/competitions/alfabank-campus/data'> kaggle.com/competitions/alfabank-campus/data </a> </br>

![data](https://github.com/DmitryTatarintsev/internship/assets/78785861/70e92e9f-47e8-4785-8055-cdf52ee4cf64)


# Подход Data Science. Конкурс классических моделей: Sklearn, Catboost, Xgboost.

![](https://github.com/DmitryTatarintsev/internship/blob/main/AlfaBankCampus/Concurs_result.png)

**Лучшая DS модель. CatBoost. Обучение.**
```python
# Создание выборок.
X_train, y_train, X_valid, y_valid, X_test, y_test = Parsing(max_lag=10, dataframe=df_train, 
                                                             rolling_mean_size=5, multi=1000).split_data()
# Создание модели CatBoostRegressor
model = CatBoostRegressor()
# Определение параметров для подбора
parameters = {
    'learning_rate': [0.01, 0.1, 0.5],
    'depth': [3, 5, 7],
    'l2_leaf_reg': [1, 3, 5]
}
# Инициализация объекта TimeSeriesSplit для кросс-валидации
tscv = TimeSeriesSplit(n_splits=3)
# Подбор параметров с использованием кросс-валидации
grid_search = GridSearchCV(estimator=model, param_grid=parameters, 
                           cv=tscv, n_jobs=4, 
                           scoring='neg_root_mean_squared_error')
```
catboost_model_v1.bin
```
0:	learn: 587.9433034	total: 164ms	remaining: 2m 43s
1:	learn: 587.4730069	total: 189ms	remaining: 1m 34s
2:	learn: 587.0081303	total: 213ms	remaining: 1m 10s
3:	learn: 586.5509548	total: 235ms	remaining: 58.4s
...
994:	learn: 554.8677960	total: 22.9s	remaining: 115ms
995:	learn: 554.8637174	total: 22.9s	remaining: 92.2ms
996:	learn: 554.8616211	total: 23s	remaining: 69.1ms
997:	learn: 554.8587688	total: 23s	remaining: 46.1ms
998:	learn: 554.8541794	total: 23s	remaining: 23ms
999:	learn: 554.8483524	total: 23s	remaining: 0us
Итоговая точность: 604.0665752020748
```

**Вторая попытка улучшить качество CatBoostRegressor.**
```python
# Создание выборок.
X_train, y_train, X_valid, y_valid, X_test, y_test = Parsing(max_lag=10, dataframe=df_train, 
                                                             rolling_mean_size=5, multi=1500).split_data()
CBR = CatBoostRegressor()
CBR.fit(X_train, y_train)
print('Итоговая точность:', mean_squared_error(y_test, CBR.predict(X_test))**.5)
```
catboost_model_v2.bin
```
Learning rate set to 0.110882
0:	learn: 599.8116149	total: 30.5ms	remaining: 30.4s
1:	learn: 595.2780258	total: 58.8ms	remaining: 29.4s
2:	learn: 591.4936683	total: 88.8ms	remaining: 29.5s
3:	learn: 588.3984841	total: 121ms	remaining: 30.1s
...
995:	learn: 556.7261328	total: 27.5s	remaining: 110ms
996:	learn: 556.7144488	total: 27.5s	remaining: 82.8ms
997:	learn: 556.7012503	total: 27.5s	remaining: 55.2ms
998:	learn: 556.6923717	total: 27.6s	remaining: 27.6ms
999:	learn: 556.6856187	total: 27.6s	remaining: 0us
Итоговая точность: 560.5032931977466
```
**Итог.**
Написали сервисные функции парсинга данных и генерации признаков. Провели конкурс моделей и победил CatBoostRegressor. 
- `catboost_model_v1.bin`. Первый этап обучения: итоговая точность - 604.1. Подбор параметров. Задействовано 1000 клиентов из 7000.
- `catboost_model_v2.bin`. Второй этап: итоговая точность - 560.5. 1500/7000 клиентов.

# Подход Deep Learning. Нейронная сеть. Keras, Tensorflow. Генератор архитектур и моделей с конкурсным отбором.
**Функция потерь.**
![](https://github.com/DmitryTatarintsev/internship/blob/main/AlfaBankCampus/best_nn_diogramma_loss.png)

![](https://github.com/DmitryTatarintsev/internship/blob/main/AlfaBankCampus/best_nn_diogramma.png)

# Заключение.

По итогам отбора, Deep Learning подход машинного обучения оказался лучше классического Data Science.
- CatBoostRegressor, RMSE: 560.5
- Нейронная сеть. Линейная Регрессионная Модель Предсказания Временного Ряда, RMSE: 513.51

Архитектура нейронной сети, лучшей модели.
```
Model: "sequential_162"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_647 (Dense)           (None, 5, 176)            352       
                                                                 
 lstm_486 (LSTM)             (None, 29)                23896     
                                                                 
 flatten_488 (Flatten)       (None, 29)                0         
                                                                 
 dense_652 (Dense)           (None, 1)                 30        
                                                                 
=================================================================
Total params: 24,278
Trainable params: 24,278
Non-trainable params: 0
_________________________________________________________________
```

```python
# Конвертация для DL данных
def convert_y_to_list(y_matrix=y_test): return [int(i) for i in yScaler.inverse_transform(y_matrix[0])]
def convert_X_to_list(X_matrix=X_test): return [int(i[0]) for i in yScaler.inverse_transform(model.predict(X_matrix[0]))]
def score(): return mean_squared_error(convert_y_to_list(y_test), convert_X_to_list(X_test))**.5

score()
```
```
3/3 [==============================] - 0s 3ms/step
513.5098365880904
```

![](https://github.com/DmitryTatarintsev/internship/blob/main/AlfaBankCampus/submission.png)
