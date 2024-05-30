"""
Módulo para el entrenamiento del modelo.
"""
import os
import numpy as np

from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.model_selection import train_test_split 

PROCESS_DATA_PATH = './processed_data'

# -----------------------------------------------------------
# Obtener las categorías y mapearlas a un número
categories = os.listdir(PROCESS_DATA_PATH)
categories_map = {category: i for i, category in enumerate(categories)}
print(categories_map)

# Obtener el número de videos y frames por video por sena
n_videos = len(os.listdir(os.path.join(PROCESS_DATA_PATH, categories[0])))
n_frames = len(os.listdir(os.path.join(PROCESS_DATA_PATH, categories[0], str(0))))

# -----------------------------------------------------------
# Crear los arrays de los datos y sus correspondientes etiquetas para entrenar el modelo

videos, labels = [], []
for category in categories:
    for video in range(n_videos):
        frames = []
        for frame_num in range(n_frames):
            info = np.load(os.path.join(PROCESS_DATA_PATH, category, str(video), str(frame_num) + '.npy'))
            frames.append(info)
        videos.append(frames)
        labels.append(categories_map[category])


x = np.array(videos)
y = to_categorical(labels).astype(int)
print(y)
print(x.shape)
print(y.shape)

# -----------------------------------------------------------
# Dividir los datos en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1) # 10% de los datos para prueba* 