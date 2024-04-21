import streamlit as st
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
    if len(img.shape) > 2: img = img[0]
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

# Преобразует EagerTensor в NumPy array
img = lambda x: Image.fromarray(x.numpy().astype(np.uint8).reshape(512, 512))

def main():
    st.title("Прогноз вероятности рассеянного склероза на снимке МРТ.")
    uploaded_file = st.file_uploader("Выберите изображение", type=["dcm", "jpg", "png"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.dcm'): img_array = read_dcm(uploaded_file.name)
        else: img_array = read_img(uploaded_file.name)     
        # Отображение выбранного изображения
        st.image(img(img_array), caption='Выбранное изображение', use_column_width=True)
        # Предсказание вероятности с помощью модели
        predictions = model.predict(img_array)
        predictions = tf.nn.softmax(predictions[0])
        predictions =  round(float(predictions[1]), 2)
        st.write(f"Вероятность: {predictions}")

if __name__ == "__main__":
    main()
