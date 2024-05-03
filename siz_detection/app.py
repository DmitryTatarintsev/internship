import streamlit as st
import os
import numpy as np
import pydicom                  # библиотека для работы с DICOM-изображениями в медицине.
import tensorflow as tf         # библиотека для создания сети

from PIL import Image           # библиотека для работы с изображениями
from tensorflow import keras
from tensorflow.keras.preprocessing import image

# Модель 
model = tf.keras.models.load_model('model.h5', compile=True)

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
    st.write("Где 0 - нет склероза, 1 - есть склероз.")
    
    # Display instruction text
    st.markdown("##### Инструкция:")
    st.write("1. Загрузите свое изображение, используя кнопку 'Upload an image'.")
    st.write("2. Либо выберите одно из предварительно загруженных изображений ниже.")
    st.write("3. После загрузки изображения результаты обработки отобразятся ниже.")
    st.write("")
    
    uploaded_file = st.file_uploader("Выберите изображение", type=["dcm", "jpg", "png"])

    if uploaded_file is None:
        # Display a row of preloaded images
        st.markdown("##### Preloaded Images")
        col1, col2 = st.columns(2)  # Create 2 columns for 2 images
        with col1:
            image1 = img(read_img("image_example_1.jpg"))
            st.image(image1, caption='jpg example', use_column_width=True)
    
        with col2:
            image2 = img(read_dcm("image_example.dcm"))
            st.image(image2, caption='dcm example', use_column_width=True)

    if uploaded_file is not None:
        if uploaded_file.name.endswith('.dcm'): img_array = read_dcm(uploaded_file)
        else: img_array = read_img(uploaded_file)     
        # Отображение выбранного изображения
        st.image(img(img_array), caption='Выбранное изображение', use_column_width=True)
        # Предсказание вероятности с помощью модели
        predictions = model.predict(img_array)
        predictions = tf.nn.softmax(predictions[0])
        predictions =  round(float(predictions[1]), 2)
        st.write(f"Вероятность: {predictions}")

    st.write("")
    st.write("Автор: https://t.me/dtatarintsev")
    st.write("GitHub проекта: https://github.com/DmitryTatarintsev/internship/tree/main/multiple_sclerosis")

if __name__ == "__main__":
    main()
