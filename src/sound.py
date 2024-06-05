"""
Este módulo utiliza la biblioteca Google Text-to-Speech (gTTS) para convertir el texto
proporcionado en habla, guarda el habla como un archivo MP3 y luego reproduce el archivo
usando la biblioteca pygame. Después de que se reproduce el habla, el archivo MP3 temporal
es eliminado.
"""
# Importar las librerías requeridas
import os
from time import sleep

from gtts import gTTS
import pygame

def text_to_speech(text:str) -> None:
    """
    Convierte el texto dado a habla y lo reproduce usando pygame.

    Args:
        text (str): El texto que se convertirá en habla.
    
    Returns:
    None
    """
    tts = gTTS(text=text, lang='es')
    filename = "speech.mp3"
    tts.save(filename)
    
    pygame.init()
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        sleep(0.5)

    pygame.mixer.quit()
    pygame.quit()

    os.remove(filename)

if __name__ == "__main__":
    texto = "Amigo"
    text_to_speech(texto)