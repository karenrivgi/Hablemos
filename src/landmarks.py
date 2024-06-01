"""
Este script contiene las funciones necesarias para mostrar los landmarks en la imagen.
"""

# Importar las librerías necesarias
import mediapipe as mp
import cv2


# Traer el modelo holístico de mediapipe
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)


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


def mostrar_landmarks(image, results=holistic):
    """
    Función para mostrar los landmarks en la imagen.

    Parameters:
        image (numpy.ndarray): La imagen de entrada en formato BGR.
        results (mediapipe.python.solutions.holistic.HolisticResults): Los resultados de la detección de MediaPipe Holistic.

    Returns:
        numpy.ndarray: La imagen con los landmarks dibujados.
    """

    image, results = mediapipe_detection(image, holistic)
    draw_styled_landmarks(image, results)
    return image