"""
Este módulo está diseñado para preparar las etiquetas de los datos para el
entrenamiento de un modelo de reconocimiento de señas. Carga los datos
procesados desde archivos .npy, organiza los datos en arrays y ajusta
las secuencias de frames para que todas tengan la misma longitud utilizando padding.
Además, guarda los datos procesados en archivos .npy para su uso posterior.
"""
# Importar las librerías necesarias
from tensorflow.keras.utils import to_categorical, pad_sequences  # type: ignore
import os
import numpy as np

# -----------------------------------------------------------

def load_processed_data(PROCESS_DATA_PATH, save_path='MP_Data', max_seq_length=15):
    """
    Carga los datos procesados desde los archivos .npy y los organiza en arrays para el entrenamiento del modelo.

    Args:
        PROCESS_DATA_PATH (str): La ruta del directorio que contiene los datos procesados organizados por categorías.
        save_path (str): La ruta donde se guardarán los archivos de datos y etiquetas.
        max_seq_length (int): Longitud máxima de las secuencias de frames. Las secuencias más cortas se rellenarán con ceros.

    Returns:
        tuple: Una tupla que contiene dos elementos:
            - x (np.ndarray): Un array de forma (n_videos, max_seq_length, num_keypoints) que contiene los datos de los videos.
            - y (np.ndarray): Un array de etiquetas en formato categórico correspondiente a cada video.
    """
    # Verificar si los archivos de datos ya existen
    x_path = os.path.join(save_path, 'x_data.npy')
    y_path = os.path.join(save_path, 'y_data.npy')
    
    # Si ya existen, los devuelve inmediatamente
    if os.path.exists(x_path) and os.path.exists(y_path):
        x = np.load(x_path)
        y = np.load(y_path)
        return x, y

    # Obtener las categorías y mapearlas a un número
    categories = os.listdir(PROCESS_DATA_PATH)
    categories_map = {category: i for i, category in enumerate(categories)}
    print(categories_map)

    # Crear los arrays de los datos y sus correspondientes etiquetas para entrenar el modelo
    videos, labels = [], []
    for category in categories:
        category_path = os.path.join(PROCESS_DATA_PATH, category)
        for video in os.listdir(category_path):
            frames = []
            video_path = os.path.join(category_path, video)
            for frame_file in os.listdir(video_path):
                info = np.load(os.path.join(video_path, frame_file))
                frames.append(info)
            videos.append(frames)
            labels.append(categories_map[category])

    # Padding de las secuencias para que todas tengan la misma longitud
    x = pad_sequences(videos, maxlen=max_seq_length, dtype='float32', padding='post', truncating='post')
    y = to_categorical(labels).astype(int)

    # Guardar los datos procesados en archivos .npy
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(x_path, x)
    np.save(y_path, y)

    return x, y


if __name__ == "__main__":
    x_train,y_train = load_processed_data('MP_Data')
    print(x_train.shape)
    print(y_train.shape)


