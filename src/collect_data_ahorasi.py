# Importar librerías necesarias
import os
import cv2
import numpy as np 
import mediapipe as mp
from helpers import mediapipe_detection, draw_styled_landmarks, extract_keypoints, there_hand

# Definición de parámetros para la recoleccion de datos
senas = np.array(['Hola', 'Amigo', 'Como Estas', 'Adios']) # Señas a interpretar
dataset_size = 5 # Número de videos a recolectar por seña


# Path for exported data, numpy arrays
DATA_PROCESSED_PATH = os.path.join('MP_Data') 

# Crear carpetas para guardar los datos recolectados
for sena in senas: 
    for sequence in range(dataset_size+1):
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

for sena in senas:
     
    count_sample = 0
    count_frame = 0
    margin_frame= 2
    min_cant_frames= 5
    frames = []
    
    while count_sample < dataset_size:
        _, frame = video.read()
        cv2.putText(frame, 'Listo para capturar la sena "{}" ({})'.format(sena, count_sample), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_8)
        image, results = mediapipe_detection(frame, holistic)

        if there_hand(results):
            count_frame += 1
            if count_frame > margin_frame: 
                cv2.putText(image, 'Capturando...', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_8)
                keypoints = extract_keypoints(results)
                frames.append(np.asarray(keypoints))

        else:
            if len(frames) > min_cant_frames + margin_frame:
                frames = frames[:-margin_frame]

                for frame_num in range(len(frames)):
                    npy_path = os.path.join(DATA_PROCESSED_PATH, sena, str(count_sample), str(frame_num))
                    np.save(npy_path, keypoints)

                    print('Se guardó:', npy_path)

                print('Secuencia guardada:', count_sample)
            
                count_sample += 1
                
            frames = []
            count_frame = 0
            cv2.putText(frame, 'Listo para capturar la sena "{}" ({})'.format(sena, count_sample), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_8)
            
        draw_styled_landmarks(image, results)
        cv2.imshow('frame', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

video.release()
cv2.destroyAllWindows()