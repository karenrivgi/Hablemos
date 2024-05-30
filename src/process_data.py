"""
Módulo correspondiente al procesamiento de las imágenes.

Se reciben videos con formato tipo .avi de la carpeta data y se
exportan los datos como archivos tipo .npy en la carpeta processed data.

Por cada vídeo se exportan 70 archivos tipo .npy. Estos archivos contienen
la información vectorizada y unidimensional de cada uno de los puntos (landmarks)
de la cara, pose, mano izquiera y mano derecha.

Si son varios vídeos de cada seña se crea la carpeta según el nombre del vídeo.
"""

# Manejo de archivos
import os

# Procesamiento de datos
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------
DATA_PATH = './data'
PROCESS_DATA_PATH = './processed_data'

# Verificar si el directorio de datos existe, si no, crearlo
if not os.path.exists(DATA_PATH):
    print(f"El directorio {DATA_PATH} no existe. Creándolo...")
    os.makedirs(DATA_PATH)

# Usar mediapipe para encontrar las landmarks 
mp_holistic = mp.solutions.holistic # Holistic model (Detectar los landmarks)
mp_drawing = mp.solutions.drawing_utils # Drawing utilities (Dibujar los landmarks)
mp_holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# De cada frame extraeremos únicamente la info importante: los landmarks o puntitos
def extract_keypoints(results):
    """
    Extrae los puntos clave (landmarks) del cuerpo, la cara y las manos
    a partir de los resultados de MediaPipe.

    Args:
        results (mediapipe.python.solutions.holistic.HolisticResults): Los resultados de la detección de MediaPipe Holistic.

    Returns:
        np.ndarray: Un array de forma (num_keypoints,) que contiene los puntos clave del cuerpo, la cara, y las manos.
                    El array tiene la siguiente estructura:
                    - Los primeros 132 valores corresponden a los 33 puntos clave del cuerpo (pose),
                    con cada punto representado por 4 valores (x, y, z, visibility).
                    - Los siguientes 1404 valores corresponden a los 468 puntos clave de la cara (face),
                    con cada punto representado por 3 valores (x, y, z).
                    - Los siguientes 63 valores corresponden a los 21 puntos clave de la mano izquierda
                    (left hand), con cada punto representado por 3 valores (x, y, z).
                    - Los últimos 63 valores corresponden a los 21 puntos clave de la mano derecha
                    (right hand), con cada punto representado por 3 valores (x, y, z).

    Ejemplo:
        results = holistic.process(image)
        keypoints = extract_keypoints(results)
    """
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# -----------------------------------------------------------
# Recorrer las carpetas con los datos
for dir_ in os.listdir(DATA_PATH):
    try:
        # Crear directorio para los datos procesados si no existe
        if not os.path.exists(os.path.join(PROCESS_DATA_PATH, str(dir_))):
            os.makedirs(os.path.join(PROCESS_DATA_PATH, str(dir_)))

        # -----------------------------------------------------------
        # Recorrer cada video en la carpeta actual
        for file_name in os.listdir(os.path.join(DATA_PATH, dir_)):
            file_path = os.path.join(DATA_PATH, dir_, file_name)

            # Crear directorio para el video si no existe
            video_dir_path = os.path.join(PROCESS_DATA_PATH, dir_, str(file_name).split('.')[0])
            os.makedirs(video_dir_path, exist_ok=True)
            
            # Leer el video
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                print(f"Error al abrir el video {file_path}")
                continue
            frame_num = 0

            # -----------------------------------------------------------
            # Recorrer cada frame del video y guardar sus keypoints
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                # Convertir el color de BGR a RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Procesar la imagen y obtener los resultados
                results = mp_holistic.process(frame)

                # Extraer los keypoints y guardarlos 
                if results is not None:
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(video_dir_path, str(frame_num))
                    np.save(npy_path, keypoints)
                    
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

                frame_num += 1
            
            cap.release()
            cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error procesando la carpeta {dir_}: {e}")
