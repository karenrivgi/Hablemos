# Manejo de archivos
import os

# Procesamiento de datos
import mediapipe as mp
import cv2

DATA_PATH = './data'

import mediapipe as mp

# Usar mediapipe para encontrar las landmarks 
mp_holistic = mp.solutions.holistic # Holistic model (Detectar los landmarks)
mp_drawing = mp.solutions.drawing_utils # Drawing utilities (Dibujar los landmarks)

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # conversion de color BGR a RGB *OpenCV tiene formato BGR
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Hacer prediccion con el modelo pasado
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_styled_landmarks(image, results):
    # Dibujar landmarks de la cara

    '''
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                             # Formato para los puntos
                             mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1), 

                             # Formato para las conexiones
                             mp_drawing.DrawingSpec(color=(0,128,0), thickness=1, circle_radius=1)
                             ) 
    
    '''
    # Dibujar landmarks de la pose
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(0,128,0), thickness=1, circle_radius=1)
                             ) 
    # Dibujar landmarks de la mano izquierda
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(0,128,0), thickness=2, circle_radius=2)
                             ) 
    # Dibujar landmarks de la mano derecha
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(0,128,0), thickness=2, circle_radius=2)
                             )

holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) # Crear modelo holistic