"""
Módulo para preparar las etiquetas de los datos.
"""
# Importar las librerías necesarias
import os
import numpy as np

# Librerias asociadas
from tensorflow.keras.utils import to_categorical # type: ignore
PROCESS_DATA_PATH = './processed_data'

# -----------------------------------------------------------
def load_processed_data(PROCESS_DATA_PATH):
    """
    Carga los datos procesados desde los archivos .npy y los organiza en arrays para el entrenamiento del modelo.

    Args:
        PROCESS_DATA_PATH (str): La ruta del directorio que contiene los datos procesados organizados por categorías.

    Returns:
        tuple: Una tupla que contiene dos elementos:
            - x (np.ndarray): Un array de forma (n_videos, n_frames, num_keypoints) que contiene los datos de los videos.
            - y (np.ndarray): Un array de etiquetas en formato categórico correspondiente a cada video.
    """
    # Obtener las categorías y mapearlas a un número
    categories = os.listdir(PROCESS_DATA_PATH)
    categories_map = {category: i for i, category in enumerate(categories)}
    print(categories_map)

    # Obtener el número de videos y frames por video por categoría
    n_videos = len(os.listdir(os.path.join(PROCESS_DATA_PATH, categories[0])))
    n_frames = len(os.listdir(os.path.join(PROCESS_DATA_PATH, categories[0], str(0))))

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

    return x, y 