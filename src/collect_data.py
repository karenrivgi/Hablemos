# Importar librerías necesarias
import os
import cv2
import numpy as np


# Definición de parámetros para la recoleccion de datos
senas = ['Hoy', 'A' ] # Señas que se van a recolectar
dataset_size = 2 # Número de videos a recolectar por seña
sequence_length = 70 # Frames de longitud de cada vídeo 


# Directorio donde se guardarán los datos
DATA_PATH = './data'
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
    
# -----------------------------------------------------------
# Inicializar la cámara
cap = cv2.VideoCapture(0)

for sena in senas:

    # Crear directorio para la seña si no existe y 
    # obtener el número máximo de archivos en el directorio

    max_number = -1 # Número máximo de archivos en el directorio
    
    if not os.path.exists(os.path.join(DATA_PATH, str(sena))):
        os.makedirs(os.path.join(DATA_PATH, str(sena)))
    else:
        file_names = os.listdir(os.path.join(DATA_PATH, sena))
        numbers = [int(file_name.split('.')[0]) for file_name in file_names]
        max_number = np.max(numbers) if numbers else -1

    print('Collecting data for class {}'.format(sena))

    # -----------------------------------------------------------
    # Esperar a que el usuario presione la tecla 'Q' para empezar
    while True:
        _, frame = cap.read()

        cv2.putText(frame, 'Presiona "Q" para empezar con la sena "{}"'.format(sena), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_8)
        cv2.imshow('frame', frame)

        if cv2.waitKey(25) == ord('q'):
            break

    # -----------------------------------------------------------
    # Capturar los videos necesarios para la seña actual
    counter = 0

    while counter < dataset_size:

        while True:
            _, frame = cap.read()

            # Mostrar un mensaje de inicio de captura
            cv2.putText(frame, 'Captura la sena "{}" ({}) presionando "C"'.format(sena, counter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_8)
            cv2.imshow('frame', frame)

            # Esperar a que el usuario presione la tecla 'C'
            if cv2.waitKey(25) == ord('c'):
                break

        
        # Crear un objeto VideoWriter para guardar el video
        video_name = '{}.avi'.format(counter + max_number + 1)
        video_path = os.path.join(DATA_PATH, sena, video_name)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

        # -----------------------------------------------------------
        # Capturar vídeo de seña
        for frame_num in range(sequence_length):
            _, frame = cap.read()
            out.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break

        counter += 1

        # Liberar el objeto VideoWriter
        out.release()


cap.release()
cv2.destroyAllWindows()