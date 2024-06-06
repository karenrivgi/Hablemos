import tkinter as tk
from src import classifier, text2sign, dictionary
from tkinter import ttk

def open_classifier_window():
    # Limpiar el frame
    for widget in main_frame.winfo_children():
        widget.destroy()

    # Aquí puedes colocar el código para abrir la ventana del clasificador (classifier.py)
    label = tk.Label(main_frame, text="Ventana del Clasificador")
    label.pack()

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

# Crear los botones
button_frame = ttk.Frame(window)
button_frame.pack(fill=tk.X)

classifier_button = tk.Button(button_frame, text="Clasificador", command=open_classifier_window)
classifier_button.pack(side=tk.LEFT)

text2sign_button = tk.Button(button_frame, text="Text2Sign", command=open_text2sign_window)
text2sign_button.pack(side=tk.LEFT)

dictionary_button = tk.Button(button_frame, text="Diccionario", command=open_dictionary_window)
dictionary_button.pack(side=tk.LEFT)

# Iniciar el bucle principal de la ventana
window.mainloop()