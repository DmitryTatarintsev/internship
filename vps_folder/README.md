![](https://github.com/salfa-ru/doct24_neural-network/blob/main/PatientsExcelData/Dmitry/png/cdn_alltend_ru_pic.jpg)

# Предсказание врожденного порока сердца у детей

<a href='https://github.com/DmitryTatarintsev/internship/blob/main/vps_folder/main.ipynb'> Итоговая модель. Deep Learning. Нейронная прогностическая модель.  </a> </br>
<a href='https://github.com/DmitryTatarintsev/internship/blob/main/vps_folder/heart_disease_classicML_model.ipynb'> Data Science. Классическое машинное обучение. Прогностическая модель CatBoostClassifier. </a> </br>
<a href='https://github.com/DmitryTatarintsev/internship/blob/main/vps_folder/Notebook.ipynb'> Применение. </a> </br>
<a href='https://github.com/DmitryTatarintsev/internship/blob/main/vps_folder/requirements.txt'> requirements.txt </a></br>

main.ipynb - проект, процесс исследования и обучения модели.</br>
Notebook.ipynb  - пример работы.</br>
model.h5  - модель.</br>
model.py -  алгоритм предобработки сырых данных для подачи в нейронную сеть. Принимает текст диагноза. Так же вызывает модель и возвращает результат.</br>
tokenizer.pickle - частный словарь.

Стэк: numpy, pandas, matplotlib, seaborn, pickle, tensorflow, sklearn.

Статус: **проект завершен.**

Цель: написать прогностическую модель для определения вероятности врожденного порока сердца у детей. Где 0 - нет порока сердца, 1 - есть порок сердца.

</br>Частота заполнения данных о пациенте.</br>
![](https://github.com/DmitryTatarintsev/internship/blob/main/vps_folder/png/df_null.png)

Результат обучения моделей нейронных сетей на разных признаках.
![](https://github.com/DmitryTatarintsev/internship/blob/main/vps_folder/png/all.png)
![](https://github.com/DmitryTatarintsev/internship/blob/main/vps_folder/png/all1.png)

В итоге, выбрали и обучили модель нейронной сети только на признаке Diagnosis
```
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (None, 400, 50)           1500000   
                                                                 
 spatial_dropout1d (SpatialD  (None, 400, 50)          0         
 ropout1D)                                                       
                                                                 
 flatten (Flatten)           (None, 20000)             0         
                                                                 
 batch_normalization (BatchN  (None, 20000)            80000     
 ormalization)                                                   
                                                                 
 dense_4 (Dense)             (None, 64)                1280064   
                                                                 
 dropout_3 (Dropout)         (None, 64)                0         
                                                                 
 batch_normalization_1 (Batc  (None, 64)               256       
 hNormalization)                                                 
                                                                 
 dense_5 (Dense)             (None, 2)                 130       
                                                                 
=================================================================
Total params: 2,860,450
Trainable params: 2,820,322
Non-trainable params: 40,128
_________________________________________________________________
```
![](https://github.com/DmitryTatarintsev/internship/blob/main/vps_folder/png/train_result.png)
![](https://github.com/DmitryTatarintsev/internship/blob/main/vps_folder/png/embedding_model.png)
</br> Верно опредленно: </br>
99% -  отсутствие порока сердца </br>
89% - порок сердца</br>
</br>
Итоговая точность прогноза модели: 94%

![](https://github.com/DmitryTatarintsev/internship/blob/main/vps_folder/png/cm_all.png)

Исследование качества прогноза других моделей того же признака с учетом погрешности.
![](https://github.com/DmitryTatarintsev/internship/blob/main/vps_folder/png/other_model_result.png)

<a href='https://github.com/DmitryTatarintsev/internship/blob/main/vps_folder/other_results.md'> Больше результатов по другим признакам </a> 
