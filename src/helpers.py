import cv2
import numpy as np
import mediapipe as mp
from typing import NamedTuple

# Traer el modelo holístico de mediapipe
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities


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
    

# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
def there_hand(results: NamedTuple) -> bool:
    return results.left_hand_landmarks or results.right_hand_landmarks


# ---------------------------------------------------------------------
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame