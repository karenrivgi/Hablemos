"""
Módulo en construcción para clasificar las señas en tiempo real.
"""


import mediapipe as mp
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from process_data import extract_keypoints
from typing import NamedTuple
from helpers import mediapipe_detection, draw_styled_landmarks, extract_keypoints, there_hand, prob_viz

# Traer el modelo holístico de mediapipe
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities


# Señas a interpretar
senas = np.array(['Hola','Amigo', 'Adios'])
model = load_model('action.h5')

# Inicializar variables
sequence = []
sentence = []
predictions = []
threshold = 0.7
colors = [(245,117,16), (117,245,16), (16,117,245)]


# ---------------------------------------------------------------------
cap = cv2.VideoCapture(0)
count_frame = 0 
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    ultimo_res = None
    
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
    
        
        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        

        if len(sequence) == 30 and there_hand(results):
            print('Hand detected')
            count_frame += 1
            print(count_frame)
            
        else:
            print('Hand not detected')
            if count_frame > 5:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                sequence = []
                print("resultado", res)

                if res[np.argmax(res)] > threshold: 
                    predictions.append(np.argmax(res))
                    ultimo_res = res
                    print("resultado threshold", res)

                    if len(sentence) > 0: 
                        if senas[np.argmax(res)] != sentence[-1]:
                            sentence.append(senas[np.argmax(res)])
                    else:
                        sentence.append(senas[np.argmax(res)])

                count_frame = 0
                sequence = []

            if len(sentence) > 5: 
                sentence = sentence[-5:]

                # Viz probabilities
        image = prob_viz(ultimo_res if ultimo_res is not None else [0, 0, 0], senas, image, colors)   
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        
        # Draw landmarks
        draw_styled_landmarks(image, results)

        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

