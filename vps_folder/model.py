def predict_proba(data):
    
    import numpy as np
    import pandas as pd
    import tensorflow
    import pickle
    
    # Предобработка
    text_data = [data]
    
    # Загружаем сохраненный Tokenizer из файла
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Преобразование входных текстов в последовательности индексов (для архитектур с embedding)
    # Применено приведение к массиву объектов для дальнейшего разделения на выборки
    x_data = np.array(tokenizer.texts_to_sequences(text_data), dtype=object)

    # Снижение размерности входных данных:
    # ограничение длины последовательностей до разумного предела
    seq_max_len = 400
    x_test_clip = tensorflow.keras.preprocessing.sequence.pad_sequences(x_data, maxlen=seq_max_len)

    model_loaded = tensorflow.keras.models.load_model('model.h5')
    pred = model_loaded.predict(x_test_clip)
    pred = tensorflow.sigmoid(pred)
    
    return np.array(pred)[0][1]