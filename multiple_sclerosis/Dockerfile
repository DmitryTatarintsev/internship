FROM python:3.10.12

RUN pip install pip==23.1.2
RUN pip install streamlit==1.33.0
RUN pip install tensorflow==2.15.0
RUN pip install numpy==1.24.3
RUN pip install pydicom==2.4.2
RUN pip install Pillow==9.4.0

# Создание рабочей директории
WORKDIR /app

COPY app.py /app/
COPY model.h5 /app/
COPY image_example.dcm /app/
COPY image_example_1.jpg /app/

# Установка переменных окружения
ENV PORT=8501

CMD ["streamlit", "run", "app.py"]

# Дополнительные команды (оставьте только те, которые нужны)
RUN apt-get update && apt-get upgrade -y
RUN apt-get update && apt-get install -y mesa-common-dev