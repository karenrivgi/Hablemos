import tkinter as tk
from tkinter import ttk
import classifier
import threading
from PIL import Image, ImageTk


def open_classifier_window():
     # Limpiar el frame
    for widget in main_frame.winfo_children():
        widget.destroy()

    # Crear un label para mostrar el video
    video_label = tk.Label(main_frame)
    video_label.pack()

    # Ejecutar el código en classifier.py en un hilo separado
    thread = threading.Thread(target=classifier.main, args=(video_label,))
    thread.start()

    # Asegurarse de que el hilo se detenga cuando se cierre la ventana
    def on_close():
        thread.join()  # Esperar a que el hilo termine
        window.destroy()  # Cerrar la ventana

    window.protocol("WM_DELETE_WINDOW", on_close)

def open_text2sign_window():
    # Limpiar el frame
    for widget in main_frame.winfo_children():
        widget.destroy()

    # Aquí puedes colocar el código para abrir la ventana de text2sign
    label = tk.Label(main_frame, text="Ventana de Text2Sign")
    label.pack()

def open_dictionary_window():
    # Limpiar el frame
    for widget in main_frame.winfo_children():
        widget.destroy()

    # Aquí puedes colocar el código para abrir la ventana del diccionario
    label = tk.Label(main_frame, text="Ventana del Diccionario")
    label.pack()

# Crear la ventana principal
window = tk.Tk()

# Crear el frame principal
main_frame = ttk.Frame(window)
main_frame.pack(fill=tk.BOTH, expand=True)

# Crear un frame para la imagen
image_frame = ttk.Frame(main_frame)
image_frame.pack()

# Crear un label para mostrar la imagen
image_label = tk.Label(image_frame)
image_label.pack()

# Cargar la imagen
image = Image.open("UI\Fondo.png")

# Redimensionar la imagen al tamaño por defecto de cv2 (640x480)
image = image.resize((640, 480))

# Convertir la imagen a formato PhotoImage de tkinter
photo = ImageTk.PhotoImage(image)

# Mostrar la imagen en el label
image_label.config(image=photo)
image_label.image = photo  # Guardar una referencia a la imagen para evitar que sea eliminada por el recolector de basura

# Crear los botones
button_frame = ttk.Frame(window)
button_frame.pack(fill=tk.X)

classifier_button = tk.Button(button_frame, text="Clasificador", command=open_classifier_window)
classifier_button.pack(side=tk.LEFT, fill=tk.X, expand=True)

text2sign_button = tk.Button(button_frame, text="Text2Sign", command=open_text2sign_window)
text2sign_button.pack(side=tk.LEFT, fill=tk.X, expand=True)

dictionary_button = tk.Button(button_frame, text="Diccionario", command=open_dictionary_window)
dictionary_button.pack(side=tk.LEFT, fill=tk.X, expand=True)

# Iniciar el bucle principal de la ventana
window.mainloop()