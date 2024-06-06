import os
import cv2

# Obtener la seña ingresada por el usuario
sena = input("Ingrese la seña que desea ver: ")

# Ruta de la carpeta de las señas
carpeta_senas = os.path.join(os.getcwd(), "Senas")

# Buscar el video correspondiente a la seña solicitada
video = None
for archivo in os.listdir(carpeta_senas):
    if sena.lower() in archivo.lower():
        video = os.path.join(carpeta_senas, archivo)
        break

# Reproducir el video si se encontró
if video:
    # Cargar el video
    cap = cv2.VideoCapture(video)

    # Verificar si el video se abrió correctamente
    if not cap.isOpened():
        print("Error al abrir el video.")
    else:
        
        # Reproducir el video 3 veces seguidas
        contador = 0
        while contador < 5:
            while True:
                # Leer un frame del video
                ret, frame = cap.read()

                # Verificar si se pudo leer el frame
                if not ret:
                    break

                # Mostrar el frame en la ventana
                cv2.imshow("Sena", frame)

                # Esperar 1 milisegundo y verificar si se presionó la tecla 'q' para salir
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break

            # Reiniciar el video al principio
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            contador += 1

        # Liberar los recursos
        cap.release()
        cv2.destroyAllWindows()
else:
    print("No se encontró el video de la seña solicitada.")