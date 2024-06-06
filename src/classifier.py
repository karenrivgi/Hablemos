"""
Módulo en construcción para clasificar las señas en tiempo real.
"""
# Importar las librerías necesarias
from tensorflow.keras.models import load_model
import mediapipe as mp
import cv2
import numpy as np
# Módulo local
from helpers import (
    mediapipe_detection, draw_styled_landmarks,
    extract_keypoints, there_hand, prob_viz, draw_text
)
from sound import text_to_speech

# Traer el modelo holístico de mediapipe
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

# Definición de parámetros para la recoleccion de datos
senas = np.array(['Amigo', 'Bien', 'Como Estas', 'Hola']) # Señas a interpretar

# Cargar el modelo entrenado
model = load_model('senas.h5')

# Inicializar variables
sequence = []        # Para almacenar las secuencias de frames
sentence = []        # Para almacenar las predicciones
predictions = []     # Para almacenar las predicciones en forma de índice

THRESHOLD = 0.7      # Umbral para considerar una predicción como válida
colors = [(245,117,16), (117,245,16), (16,117,245), (0,255,0)]  # Colores para la visualización

# Inicializar la captura de video
cap = cv2.VideoCapture(0)
COUNT_FRAME = 0

# Establecer el modelo de mediapipe
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    ultimo_res = None  # Variable para almacenar la última predicción

    while cap.isOpened():
        # Leer el feed de la cámara
        ret, frame = cap.read()

        # Hacer detecciones utilizando mediapipe
        image, results = mediapipe_detection(frame, holistic)
        
        # Lógica de predicción
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-20:]  # Mantener solo los últimos 30 frames

        # Si se han capturado 20 frames y hay una mano detectada
        if len(sequence) == 20 and there_hand(results):
            COUNT_FRAME += 1
            print(COUNT_FRAME)
        else:
            if COUNT_FRAME > 5:
                # Realizar la predicción
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                sequence = []

                if res[np.argmax(res)] > THRESHOLD: 
                    predictions.append(np.argmax(res))
                    ultimo_res = res
                    print(senas[np.argmax(res)])
                    print("resultado THRESHOLD", res)
                    # Traducción de texto a voz
                    text_to_speech(senas[np.argmax(res)])

                    # Actualizar la oración con la nueva predicción
                    if len(sentence) > 0: 
                        if senas[np.argmax(res)] != sentence[-1]:
                            sentence.append(senas[np.argmax(res)])
                    else:
                        sentence.append(senas[np.argmax(res)])

                    print(dict(enumerate(reversed(sentence))))

                COUNT_FRAME = 0
                sequence = []

            # Mantener solo las últimas 5 predicciones en la oración
            if len(sentence) > 4: 
                sentence = sentence[-4:]

        # Visualizar las probabilidades
        image = prob_viz(ultimo_res if ultimo_res is not None else [0, 0, 0, 0], senas, image, colors)
        
        # Visualizar las últimas 4 detecciones
        image = draw_text(image, sentence)

        # Dibujar los landmarks
        draw_styled_landmarks(image, results)

        # Mostrar la imagen en la pantalla
        cv2.imshow('OpenCV Feed', image)

        # Salir del bucle si se presiona la tecla 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Liberar la cámara y cerrar las ventanas
    cap.release()
    cv2.destroyAllWindows()
