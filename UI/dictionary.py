"""
Módulo para habilitar un diccionario de lengua de señas en la 
interfaz gráfica de usuario (GUI) con tkinter.
"""

# Importar las librerías requeridas
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import os
from threading import Thread


# -----------------------------------------------------------
def play_video(video_path, label):
    """
    Reproduce un video en un label de tkinter.

    Args:
        video_path (str): Ruta del video a reproducir.
        label: El widget Label de tkinter donde se mostrará el video.
    Returns:
        None
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return


    # Función para actualizar el frame en el label con la extensión del vídeo
    def update_frame():
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (200, 150)) 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            label.imgtk = imgtk
            label.configure(image=imgtk)
            label.after(25, update_frame)
        else:
            cap.release()

    # Iniciar la reproducción del vídeo
    update_frame()


# -----------------------------------------------------------
def show_videos(frame, category):
    """
    Muestra los vídeos de una categoría en el frame especificado.

    Busca los vídeos en la carpeta correspondiente a la categoría y crea un
    sub-frame para cada vídeo. Cada sub-frame contiene una imagen del primer
    frame del vídeo, el título del vídeo y un botón para reproducir el vídeo.

    Args:
        frame: El frame de tkinter donde se mostrarán los vídeos.
        category (str): La categoría de los vídeos a mostrar.

    Returns:
        None 
    """
    # Limpiar el frame existente
    for widget in frame.winfo_children():
        widget.destroy()

    # Verificar si existen vídeos para la categoría
    category_folder = os.path.join("UI\Senas", category)
    if os.path.isdir(category_folder):
        pass
    else:
        help_frame = tk.Frame(frame, bg="#F5B7B1")
        help_label = tk.Label(help_frame, text="¿No encuentras la seña? Dale click\nal botón para realizar una petición.", bg="#F5B7B1")
        help_button = tk.Button(help_frame, text="¡Vamos!", bg="#FFE34C", fg="white")
        help_frame.pack(pady=10)
        help_label.pack(pady=5)
        help_button.pack(pady=5)
        return

    # Almacenar en una lista la ruta de los vídeos existentes en la carpeta
    videos = []
    for video in os.listdir(category_folder):
        video_path = os.path.join(category_folder, video)
        videos.append((video, video_path))

    # Crear el frame para los vídeos
    videos_frame = tk.Frame(frame, bg="#E6E6FA")
    videos_frame.pack(pady=10)

    row = 0
    col = 0
    
    # Crear un sub-frame para cada vídeo
    for title, video_path in videos:
        
        video_frame = tk.Frame(videos_frame, bg="#FFFFFF", borderwidth=2, relief="groove")
        video_frame.grid(row=row, column=col, padx=10, pady=10)

        # Capturar el primer frame del vídeo
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if ret:
            # Convertir el frame a un formato que Tkinter pueda mostrar
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = frame.resize((200, 150))
            frame = ImageTk.PhotoImage(frame)

            # Crear un label para mostrar el frame
            frame_label = tk.Label(video_frame, image=frame)
            frame_label.image = frame  # Guardar una referencia a la imagen 
            frame_label.pack()

        # Etiqueta de título del vídeo
        label = tk.Label(video_frame, text=title, bg="#FFFFFF")
        label.pack()

        # Botón para reproducir el vídeo
        play_button = tk.Button(video_frame, text="Play", command=lambda p=video_path, l=frame_label: Thread(target=play_video, args=(p, l)).start())
        play_button.pack()

        # Actualizar las posiciones de las columnas y filas
        col += 1
        if col > 1:
            col = 0
            row += 1


# -----------------------------------------------------------
def main(frame):
    """
    Función principal para mostrar el diccionario de lengua de señas en la GUI.

    Crea un campo de búsqueda, una lista de categorías y un botón de ayuda para
    realizar una petición de una seña no encontrada.

    Args:
        frame: El frame de tkinter donde se mostrará el diccionario.
    Returns:
        None
    """
    frame.config(width=640, height=480)  # Configurar tamaño del frame
    frame.pack_propagate(0)  

    # Texto de ayuda y botón
    help_frame = tk.Frame(frame, bg="#F5B7B1")
    help_label = tk.Label(help_frame, text="¿No encuentras la seña? Dale click\nal botón para realizar una petición.", bg="#F5B7B1")
    help_button = tk.Button(help_frame, text="¡Vamos!", bg="#FFE34C", fg="white")
    help_label.pack(pady=5)
    help_button.pack(pady=5)
    help_frame.pack(pady=10)

    # Crear el frame para las categorías
    categories_frame = tk.Frame(frame)
    categories_frame.pack(pady=10)

    # Lista de categorías con sus imágenes
    categories = [
        ("Saludos", "UI\Imagenes\Saludos.png"),
        ("Relaciones", "UI\Imagenes\Relaciones.png"),
        ("Viajes", "UI\Imagenes\Viajes.png"),
    ]

    # Crear los botones de categorías
    row = 0
    col = 0

    for category, icon_path in categories:

        img = Image.open(icon_path)
        img = img.resize((110, 80)) 
        image = ImageTk.PhotoImage(img)

        button = tk.Button(categories_frame, image=image, compound="top", bg="#D7BDE2",
                       command=lambda category=category: show_videos(frame, category))
        
        button.image = image  # Guardar referencia de la imagen
        button.command = command=lambda: show_videos(frame, category)
        button.grid(row=row, column=col, padx=10, pady=10)

        col += 1
        if col > 1:
            col = 0
            row += 1