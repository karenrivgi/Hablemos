# Importar las librerías necesarias
from tensorflow.keras.models import load_model
import mediapipe as mp
import cv2
import numpy as np
from PIL import Image, ImageTk
# Módulo local
from helpers import (
    mediapipe_detection, draw_styled_landmarks,
    extract_keypoints, there_hand, prob_viz, draw_text
)

def main(label):
    cap = cv2.VideoCapture(0)

    # Traer el modelo holístico de mediapipe
    mp_holistic = mp.solutions.holistic # Holistic model

    # Definición de parámetros para la recoleccion de datos
    senas = np.array(['Amigo', 'Bien', 'Como estas?', 'Hola']) # Señas a interpretar

    # Cargar el modelo entrenado
    model = load_model('senas.h5')

    # Inicializar variables
    sequence = []        # Para almacenar las secuencias de frames
    sentence = []        # Para almacenar las predicciones
    THRESHOLD = 0.7      # Umbral para considerar una predicción como válida

    try:
        COUNT_FRAME = 0
        # Establecer el modelo de mediapipe
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

            while cap.isOpened():
                # Leer el feed de la cámara
                _, frame = cap.read()

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
                            print(senas[np.argmax(res)])
                            print("resultado THRESHOLD", res)

                            # Actualizar la lista de predicciones con la nueva predicción
                            if len(sentence) > 0: 
                                if senas[np.argmax(res)] != sentence[-1]:
                                    sentence.append(senas[np.argmax(res)])
                            else:
                                sentence.append(senas[np.argmax(res)])

                            print(dict(enumerate(reversed(sentence))))

                        COUNT_FRAME = 0
                        sequence = []

                    # Mantener solo las últimas 4 predicciones
                    if len(sentence) > 4: 
                        sentence = sentence[-4:]
                
                # Visualizar las últimas 4 detecciones
                image = draw_text(image, sentence)

                # Dibujar los landmarks
                draw_styled_landmarks(image, results)

                # Convertir la imagen con los landmarks a una imagen de PhotoImage
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                image = ImageTk.PhotoImage(image)

                # Mostrar la imagen en el label
                label.config(image=image)
                label.image = image

                # Salir del bucle si se presiona la tecla 'q'
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    finally:

        # Cuando todo esté hecho, liberar la captura
        cap.release()
        cv2.destroyAllWindows()