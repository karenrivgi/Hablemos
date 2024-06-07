"""
Módulo para la reproducción de videos en una interfaz gráfica de usuario (GUI) con tkinter.

Este módulo proporciona funciones para reproducir videos en una ventana de tkinter.
"""

# Importar las librerías requeridas
import os
import cv2
from PIL import Image, ImageTk


def search(search_var:str, video_label) -> None:
    """
    Busca el video correspondiente a la seña ingresada por el usuario
    y lo reproduce si se encuentra.

    Args:
        search_var (tk.StringVar): Variable de cadena asociada a la barra de
        búsqueda donde el usuario ingresa la seña.

        video_label (tk.Label): Label donde se mostrará el video.

    Returns:
        None
    """
    user_input = search_var.get()  # Obtener la seña ingresada por el usuario
    video_path = find_video(user_input)  # Buscar el video correspondiente
    if video_path:
        play_video(video_path, video_label)  # Reproducir el video si se encuentra
    else:
        print(f"No se encontró el video para la seña: {user_input}")  # Mensaje de error si no se encuentra el video


def find_video(sena:str) -> None:
    """
    Busca el video correspondiente a la seña en la carpeta "Senas".

    Args:
        sena (str): La seña ingresada por el usuario.

    Returns:
        str or None: La ruta completa del video si se encuentra, None si no se encuentra.
    """
    carpeta_senas = os.path.join(os.getcwd(), "Senas")  # Ruta de la carpeta de las señas
    for archivo in os.listdir(carpeta_senas):
        if sena.lower() in archivo.lower():  # Verificar si el nombre del archivo contiene la seña
            return os.path.join(carpeta_senas, archivo)  # Devolver la ruta completa del video
    return None  # Devolver None si no se encuentra el video


def play_video(video_path:str, label,
               width=500, height=380, max_repeats=3) -> None:
    """
    Reproduce un video en un label de tkinter.

    Args:
        video_path (str): Ruta del video a reproducir.
        label: El widget Label de tkinter donde se mostrará el video.
        width (int): Ancho deseado del video.
        height (int): Alto deseado del video.
        max_repeats (int): Número máximo de veces que se repetirá el video.

    Returns:
        None
    """
    cap = cv2.VideoCapture(video_path)
    repeat_count = 0

    def update_frame():
        nonlocal repeat_count
        ret, frame = cap.read()
        if ret:
            # Redimensionar el frame
            frame = cv2.resize(frame, (width, height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            label.imgtk = imgtk
            label.configure(image=imgtk)
            label.after(25, update_frame)
        else:
            if repeat_count < max_repeats - 1:
                repeat_count += 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                label.after(10, update_frame)
            else:
                cap.release()

    update_frame()


