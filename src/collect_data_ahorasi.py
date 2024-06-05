"""
Este módulo está diseñado para capturar secuencias de video desde una cámara
en tiempo real, procesarlas para detectar puntos clave (landmarks)
usando la biblioteca Mediapipe, y almacenar estos datos procesados para su
posterior uso en el módulo train_model.py.

El módulo se inicia capturando video desde la cámara y mostrando un
mensaje en el frame. A medida que se capturan y procesan frames,
se detectan los puntos clave y se almacenan los datos procesados en archivos .npy.
El proceso se repite para cada señal definida en la lista senas.
"""
# Importar librerías necesarias
import os
import cv2
import numpy as np 
import mediapipe as mp

# Modulo local
from helpers import (
     mediapipe_detection, draw_styled_landmarks,
     extract_keypoints, there_hand
     )

# Definición de parámetros para la recoleccion de datos
senas = np.array(['Hola', 'Amigo', 'Bien', 'Como Estas']) # Señas a interpretar

DATASET_SIZE= 30 # Número de videos a recolectar por seña


# Path for exported data, numpy arrays
DATA_PROCESSED_PATH = os.path.join('MP_Data') 

# Crear carpetas para guardar los datos recolectados
for sena in senas: 
    for sequence in range(DATASET_SIZE):
        try: 
            os.makedirs(os.path.join(DATA_PROCESSED_PATH, sena, str(sequence)))
        except:
            pass

# -----------------------------------------------------------
# Importar el modelo holístico de mediapipe
holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# -----------------------------------------------------------
# Inicializar la cámara
video = cv2.VideoCapture(0)

# Iterar sobre cada señal que se desea capturar
for sena in senas:
    
    # Inicializar contadores y listas
    COUNT_SAMPLE = 0
    COUNT_FRAME = 0
    MARGIN_FRAME = 2
    MIN_CANT_FRAMES = 5
    frames = []

    # Mientras no se hayan capturado todas las muestras necesarias
    while COUNT_SAMPLE < DATASET_SIZE :
        # Leer un frame de la cámara
        _, frame = video.read()
        
        # Mostrar mensaje en el frame indicando la señal a capturar
        cv2.putText(frame, 'Listo para capturar la sena "{}" ({})'.format(sena, COUNT_SAMPLE), 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_8)
        
        # Procesar el frame con Mediapipe para detectar puntos clave
        image, results = mediapipe_detection(frame, holistic)

        # Si se detectan manos en el frame
        if there_hand(results):
            COUNT_FRAME += 1
            # Si se ha superado el margen de frames inicial
            if COUNT_FRAME > MARGIN_FRAME: 
                # Mostrar mensaje de captura en el frame
                cv2.putText(image, 'Capturando...', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                            (0, 255, 255), 2, cv2.LINE_8)
                
                # Extraer puntos clave y agregarlos a la lista de frames
                keypoints = extract_keypoints(results)
                frames.append(np.asarray(keypoints))
        else:
            # Si se han capturado suficientes frames válidos más el margen
            if len(frames) > MIN_CANT_FRAMES + MARGIN_FRAME:
                # Eliminar los últimos frames de margen
                frames = frames[:-MARGIN_FRAME]

                # Guardar cada frame como un archivo .npy
                for frame_num in range(len(frames)):
                    npy_path = os.path.join(DATA_PROCESSED_PATH, sena, str(COUNT_SAMPLE), str(frame_num))
                    np.save(npy_path, keypoints)
                    print('Se guardó:', npy_path)

                print('Secuencia guardada:', COUNT_SAMPLE)
                
                # Incrementar el contador de muestras capturadas
                COUNT_SAMPLE += 1
                
            # Reiniciar lista de frames y contador de frames
            frames = []
            COUNT_FRAME = 0
            # Mostrar mensaje de listo para capturar en el frame
            cv2.putText(frame, 'Listo para capturar la sena "{}" ({})'.format(sena, COUNT_SAMPLE), 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_8)
        
        # Dibujar puntos clave estilizados en el frame
        draw_styled_landmarks(image, results)
        # Mostrar el frame procesado en una ventana
        cv2.imshow('frame', image)

        # Salir del bucle si se presiona la tecla 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Liberar la captura de video y cerrar las ventanas de OpenCV
video.release()
cv2.destroyAllWindows()