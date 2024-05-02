![](https://github.com/DmitryTatarintsev/internship/blob/main/multiple_sclerosis/img/c6e5c47311949440d1fee4d8c6df71bf.jpg)

# Предсказание рассеянного склероза
<a href='https://github.com/DmitryTatarintsev/internship/blob/main/multiple_sclerosis/make_datasets.ipynb'> Подготовка данных </a> </br>
<a href='https://github.com/DmitryTatarintsev/internship/blob/main/multiple_sclerosis/all_experiments.ipynb'> Разные эксперименты. Тетрадь </a> </br>
<a href='https://github.com/DmitryTatarintsev/internship/blob/main/multiple_sclerosis/main.ipynb'> Прогностическая модель. Тетрадь </a> </br>
<a href='https://github.com/DmitryTatarintsev/internship/blob/main/multiple_sclerosis/app.py'> Алгоритм веб-приложения. </a> </br>
<a href='https://github.com/DmitryTatarintsev/internship/blob/main/multiple_sclerosis/model.h5'> Итоговый h5 файл модели. </a> </br>
<a href='https://github.com/DmitryTatarintsev/internship/blob/main/multiple_sclerosis/requirements.txt'> requirements.txt </a> </br>

Стэк: numpy, pandas, matplotlib, seaborn, tensorflow, sklearn.

Статус: **Завершен.**

Веб-приложение (Демо версия): [Прогностическая модель рассеянного склероза по МРТ снимкам. РАБОЧАЯ ССЫЛКА](https://huggingface.co/spaces/dmitry212010/sclerosis_streamlit)

Цель: написать прогностическую модель для определения вероятности рассеянного склероза. Где 0 - нет склероза, 1 - есть склероз.

Набор данных: <a href='https://mosmed.ai/en/datasets/aie21selftestmri/'> mosmed.ai </a> </br>

```
PAN_BOT_anon             22
ING15_GB3_anon           24
INTERA_GKB12_anon        25
MRI84653_anon            29
EXCELMRI_GP22_anon      369
EXCELMRI_DC3_anon       481
EXCELMRI_GP8_anon       536
EXCELMRI_GVV2_anon      681
EXCELMRI_GKB71_anon     691
EXCELMRI_GKB57_anon     718
EXCELMRI_GP9_anon       818
HDX_MR_anon             828
EXCELMR_GB13_anon       830
EXCELMR_VERES_anon      832
EXCELMRI_GP134_anon     885
EXCELMRI_GKB36_anon     957
EXCELMRI_GP19_anon      967
EXCELMRI_GP209_anon    1086
EXCELMRI_GP212_anon    1154
EXCELMRI_GP214_anon    1204
EXCELMRI_GKB50_anon    1226
EXCELMRI_GP195_anon    1302
EXCELMRI_KDC4_anon     1306
EXCELMRI_GVV3_anon     1315
MR_GP46F1_anon         1339
EXCELMRI_GP52_anon     1359
EXCELMRI_GP67_anon     1392
EXCELMRI_GP68_anon     1574
EXCELMRI_MUH_anon      1633
EXCELMR1_GB1_anon      1644
EXCELMR_NPPCS_anon     1717
EXCELMRI_GP45_anon     1718
EXCELMRI_GP6_anon      1764
EXCELMRI_GP220_anon    1816
EXCELMR_YUD_anon       1833
EXCELMRI_GP2_anon      1947
EXCELMRI_KDC6_anon     2035
EXCELMRI_SM4_anon      2102
EXCELMR2_GB1_anon      2113
EXCELMRI_GVV1_anon     2312
ING30_MDGKB_anon       2410
EXCELMRI_GKB29_anon    2873
Name: major_dir, dtype: int64

labels.xlsx

        study_uid	                                        sclerosis
0	1.2.643.5.1.13.13.12.2.77.8252.011411090410101...	0
1	1.2.643.5.1.13.13.12.2.77.8252.131500041112120...	1
2	1.2.643.5.1.13.13.12.2.77.8252.010305060009140...	0
3	1.2.643.5.1.13.13.12.2.77.8252.071406031309010...	1
4	1.2.643.5.1.13.13.12.2.77.8252.121511041410151...	0
...	...	...
167	1.2.643.5.1.13.13.12.2.77.8252.111211060701140...	1
168	1.2.643.5.1.13.13.12.2.77.8252.100815141103051...	0
169	1.2.643.5.1.13.13.12.2.77.8252.060804010900090...	1
170	1.2.643.5.1.13.13.12.2.77.8252.030410101303070...	0
171	1.2.643.5.1.13.13.12.2.77.8252.040811010609121...	1
172 rows × 2 columns
```
В наборе данных есть фотографии разных размеров. Вот список их размеров shape в списке [(384, 416), (748, 640), (384, 384), (512, 512), (384, 384), (768, 768), (288, 288), (640, 640), (384, 384), (384, 384), (768, 768), (22, 192, 192), (384, 384), (768, 768), (640, 640), (512, 512), (768, 768), (384, 384), (576, 576), (384, 384), (384, 384), (768, 768), (768, 768), (384, 384), (768, 768), (576, 576), (23, 192, 192), (48, 128, 128), (512, 512), (25, 192, 192), (768, 768), (192, 192), (384, 384), (384, 384), (384, 384), (512, 512), (640, 640), (768, 768), (512, 512), (640, 640), (384, 384), (768, 768)]

Всего 43 папки. Внутри каждой 4 исследования. Внутри каждого исследования папки с DICOM-изображениями. Общий объем 51 Гб. Предаврительно оставили только поперечные разрезы (продольные не нужны). labels.xlsx файл содержит информацию - результат исследования, наличие или отсутствие склероза.

Определили какой dcm к какому классу относится. Написали код на основе таблицы labels.xlsx, заменили названия исследований на названия DICOM-изображений. И сохранили в новой таблице labels_plus.xlsx. 
Масштабировали DICOM-изображения для того, чтобы привести значения пикселей к одному диапазону значений и уменьшить влияние шумов на изображение.
Далее распределили dcm по выборкам и сгенерировали на их основе изображения одного размера. Сохранили изображения по соответсвующим каталогам. Сформировали датасет из каталога функцией кераса image_dataset_from_directory и приступили к обучению.

```
labels_plus.xlsx

        path	                                                sclerosis sample
0	mosmed_ai_datasets\EXCELMRI_DC3_anon\1.2.643.5...	0	  train
1	mosmed_ai_datasets\EXCELMRI_DC3_anon\1.2.643.5...	0	  train
2	mosmed_ai_datasets\EXCELMRI_DC3_anon\1.2.643.5...	0	  train
3	mosmed_ai_datasets\EXCELMRI_DC3_anon\1.2.643.5...	0	  train
4	mosmed_ai_datasets\EXCELMRI_DC3_anon\1.2.643.5...	0	  train
...	...	...	...
51862	mosmed_ai_datasets\EXCELMRI_GKB50_anon\1.2.643...	1	  train
51863	mosmed_ai_datasets\EXCELMRI_GKB50_anon\1.2.643...	1	  test
51864	mosmed_ai_datasets\EXCELMRI_GKB50_anon\1.2.643...	1	  train
51865	mosmed_ai_datasets\EXCELMRI_GKB50_anon\1.2.643...	1	  test
51866	mosmed_ai_datasets\EXCELMRI_GKB50_anon\1.2.643...	1	  train
51867 rows × 3 columns
```

```python
# Создание набора данных из изображений в каталоге
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='binary',
    color_mode='grayscale',
    batch_size=32,
    image_size=(512, 512),
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset='training')

# Создание набора данных для валидации
val_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='binary',
    color_mode='grayscale',
    batch_size=32,
    image_size=(512, 512),
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset='validation')

# Создание набора данных для теста
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='binary',
    color_mode='grayscale',
    batch_size=32,
    image_size=(512, 512),
    shuffle=True,
    seed=42)

# Сохраняем датасеты
tf.data.Dataset.save(train_dataset, tofolder)
tf.data.Dataset.save(val_dataset, tofolder)
tf.data.Dataset.save(test_dataset, tofolder)
```
```
Found 41493 files belonging to 2 classes.
Using 33195 files for training.
Found 41493 files belonging to 2 classes.
Using 8298 files for validation.
Found 10374 files belonging to 2 classes.
```

![](https://github.com/DmitryTatarintsev/internship/blob/main/multiple_sclerosis/img/subplot.png)

### Прогностическая модель
```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 510, 510, 32)      320       
                                                                 
 max_pooling2d (MaxPooling2D  (None, 170, 170, 32)     0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 168, 168, 64)      18496     
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 56, 56, 64)       0         
 2D)                                                             
                                                                 
 conv2d_2 (Conv2D)           (None, 54, 54, 128)       73856     
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 18, 18, 128)      0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 41472)             0         
                                                                 
 dense (Dense)               (None, 128)               5308544   
                                                                 
 dropout (Dropout)           (None, 128)               0         
                                                                 
 dense_1 (Dense)             (None, 2)                 258       
                                                                 
=================================================================
Total params: 5,401,474
Trainable params: 5,401,474
Non-trainable params: 0
_________________________________________________________________
```

![](https://github.com/DmitryTatarintsev/internship/blob/main/multiple_sclerosis/img/acc_loss.png)

![](https://github.com/DmitryTatarintsev/internship/blob/main/multiple_sclerosis/img/cm.png)

### Программа. Алгоритм предобработки и прогноза
Программа принимает путь к рентгеновскому снимку (jpg, dicom) и возвращает вероятность (от 0 до 1) рассеянного склероза.
```python
text = '''
def predict_proba(path):
    
    import os
    import numpy as np
    import pydicom                  # библиотека для работы с DICOM-изображениями в медицине.
    import tensorflow as tf         # библиотека для создания сети

    from PIL import Image           # библиотека для работы с изображениями
    from tensorflow import keras
    from tensorflow.keras.preprocessing import image

    # Модель 
    model = tf.keras.models.load_model('model.h5')

    # функция для принятия на вход dcm формат        
    def read_dcm(path):
        # Конвертируем
        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array.astype(np.float32) # преобразование изображения в numpy-массив
        intercept = dcm.RescaleIntercept if 'RescaleIntercept' in dcm else 0.0
        slope = dcm.RescaleSlope if 'RescaleSlope' in dcm else 1.0
        img = slope * img + intercept # масштабирование
        if len(img.shape) > 2:
            img = img[0]
        img -= img.min()
        img /= img.max()
        img *= 255.0
        img = img.astype('uint8')

        img = Image.fromarray(img).convert('L') # Преобразование в изображение в оттенках серого
        img = img.resize((512, 512)) # Изменение размера изображения
        img = image.img_to_array(img)
        return tf.expand_dims(img, 0)  # добавляем дополнительное измерение (batch size)

    def read_img(path):
        img = image.load_img(path, color_mode='grayscale', target_size=(512, 512))
        img_array = image.img_to_array(img)
        return tf.expand_dims(img_array, 0)  # добавляем дополнительное измерение (batch size)

    def predict_proba(path):
        if os.path.isfile(path):
            if path.endswith('.dcm'):
                img_array = read_dcm(path)
            elif path.endswith('.jpg'):
                img_array = read_img(path)
            else:
                print('Unsupported file extension')
        else:
            print('File not found')

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        return round(float(score[1]), 2)

    return predict_proba(path)

'''
# сохранение
with open('model.py', 'w+', encoding="utf-8",) as f:  
    f.write(text.strip())
```
### Применение

```python
import model
model.predict_proba(""example/image_example.dcm")
```
```
1/1 [==============================] - 0s 120ms/step
0.59
```

```python
import model
model.predict_proba(""example/image_example_1.jpg")
```
```
1/1 [==============================] - 0s 68ms/step
0.27
```

### Вывод

Изначально создал архитектуру:
```python
# Сверточный слой с 32 фильтрами и размером ядра (3,3).
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
# Пулинг слой для уменьшения размерности изображения.
model.add(MaxPooling2D((5, 5)))
# Сверточный слой с 32 фильтрами и размером ядра (3,3).
model.add(Conv2D(64, (3, 3), activation='relu'))
# Пулинг слой для уменьшения размерности изображения.
model.add(MaxPooling2D((5, 5)))
# Сверточный слой с 32 фильтрами и размером ядра (3,3).
model.add(Conv2D(128, (3, 3), activation='relu'))
# Пулинг слой для уменьшения размерности изображения.
model.add(MaxPooling2D((5, 5)))

# Блок классификации
# Слой преобразования многомерных данных в одномерные 
model.add(Flatten())
# Полносвязный слой с 128 нейронами и функцией активации relu.
model.add(Dense(128, activation='relu'))
# Слой Dropout для регуляризации модели.
model.add(Dropout(0.5))
# Выходной слой с одним нейроном и функцией активации sigmoid.
model.add(Dense(2, activation='sigmoid'))
```
А выборки в image_dataset_from_directory разбивал батчами - 32.  

Идея была в том, что бы собрать мобильную быструю архитектуру с небольшим числом параметров (250 000), которая в сверточных слоях сжимала изображение 512x512 пикселей до 3x3. Которая пронеслась бы по датасету размером в 52 тысячи снимков. Модель обучалась 5 часов и достигла точности 79%. 

Было принято решение пере разбить выборки в image_dataset_from_directory с батчами - 16. В архитектуре MaxPooling2D заменить сжатие с 5,5 до 3,3. Тем самым сжимать изображение в сверточных слоях не до 3x3, а до 64x64. Заменил активаторы с relu на более чувствительные elu. Все это привело к увеличению числа параметров с 250 тысяч до 5 миллионов, скорость обучения значительно уменьшилась, а качество выросло. В итоге, после обучения в два захода (в обще сумме 13 часов), достигли точности 94%.

Результатом удовлетворен, изменений не планируется.
