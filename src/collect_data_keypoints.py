"""
Módulo para colectar los datos para el modelo.

Depeniendo de la cantidad de señas y el número de señas a realizar 
se exportan en una carpeta data con sub carpetas para cada seña.
"""

# Importar librerías necesarias
import os
import cv2
import numpy as np 
import mediapipe as mp
from process_data import extract_keypoints

# Definición de parámetros para la recoleccion de datos

senas = np.array(['Hola', 'Amigo', 'Adios']) # Señas a interpretar
dataset_size = 30 # Número de videos a recolectar por seña
sequence_length = 30 # Frames de longitud de cada vídeo 


# Path for exported data, numpy arrays
DATA_PROCESSED_PATH = os.path.join('MP_Data') 

"""
for sena in senas: 
    for sequence in range(dataset_size+1):
        try: 
            os.makedirs(os.path.join(DATA_PROCESSED_PATH, sena, str(sequence)))
        except:
            pass
"""

# Traer el modelo holístico de mediapipe
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ---------------------------------------------------------------------
def mediapipe_detection(image, model):
    """
    Función para hacer una predicción con el modelo de MediaPipe.

    Parameters:
        image (numpy.ndarray): La imagen de entrada en formato BGR.
        model (MediaPipeModel): El modelo de MediaPipe utilizado para hacer la predicción.

    Returns:
        tuple: Una tupla que contiene la imagen procesada en formato BGR y los resultados de la predicción.
    """

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results


# ---------------------------------------------------------------------
def draw_styled_landmarks(image, results):
    """
    Dibuja los landmarks en la imagen de entrada.

    Parameters:
        - image: La imagen de entrada para dibujar los landmarks.
        - results: Los resultados de la predicción de MediaPipe.

    Returns:
        None
    """
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )



# -----------------------------------------------------------
# Inicializar la cámara
cap = cv2.VideoCapture(0)

for sena in senas:

    # -----------------------------------------------------------
    # Esperar a que el usuario presione la tecla 'Q' para empezar
    while True:

        # Leer el frame de la cámara
        _, frame = cap.read()

        cv2.putText(frame, 'Presiona "Q" para empezar con la sena "{}"'.format(sena), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_8)
        
        # Detectar los landmarks en el frame
        image, results = mediapipe_detection(frame, holistic)
        # Dibujar los landmarks
        draw_styled_landmarks(image, results)
        
        # Mostrar imagen con landmarks
        cv2.imshow('frame', image)

        # Si el usuario presiona la tecla 'X', cerrar la cámara
        if cv2.waitKey(10) & 0xFF == ord('x'):
                cap.release()
                cv2.destroyAllWindows()

        # Si el usuario presiona la tecla 'Q', empezar a capturar los videos
        if cv2.waitKey(25) == ord('q'):
            break
        

    # -----------------------------------------------------------
    # Capturar los videos necesarios para la seña actual
    counter = 0
    # CAMBIAR LA VARIABLE SEGÚN LO DICHO POR EL GRUPO
    # Gabriel: 0
    # Karen: 10
    # Luisa: 20
    # Felipe: 30
    # Mateo: 40
    inicial = 0

    while counter < dataset_size:

        while True:
            _, frame = cap.read()

            # Mostrar un mensaje de inicio de captura
            cv2.putText(frame, 'Captura la sena "{}" ({}) presionando "C"'.format(sena, inicial), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_8)
            
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            cv2.imshow('frame', image)

            # Si el usuario presiona la tecla 'X', cerrar la cámara
            if cv2.waitKey(10) & 0xFF == ord('x'):
                cap.release()
                cv2.destroyAllWindows()

            # Esperar a que el usuario presione la tecla 'C' para capturar el video
            if cv2.waitKey(25) == ord('c'):
                break


        # -----------------------------------------------------------
        # Capturar los frames necesaros para cada vídeo
        for frame_num in range(sequence_length):
            _, frame = cap.read()

            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            cv2.imshow('frame', image)

            # Extraer los keypoints de los landmarks detectados y guardarlos.
            keypoints = extract_keypoints(results)
            npy_path = os.path.join(DATA_PROCESSED_PATH, sena, str(counter), str(frame_num))
            np.save(npy_path, keypoints)
            
            # Si el usuario presiona la tecla 'X', cerrar la cámara
            if cv2.waitKey(10) & 0xFF == ord('x'):
                cap.release()
                cv2.destroyAllWindows()

        counter += 1
        inicial += 1


cap.release()
cv2.destroyAllWindows()
