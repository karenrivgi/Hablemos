# Librerias estandas
import os
import numpy as np
# Librerias asociadas
from sklearn.model_selection import train_test_split 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
# Modulo local
from train_classifier import load_processed_data

# Señas a interpretar
senas = np.array(['Hola', 'Buenos dias', 'Como estas', 'Amigo', 'Adios'])

# Path de los datos apra el modelo
PROCESS_DATA_PATH = './processed_data'

# Obtener los datos para el modelo
x, y = load_processed_data(PROCESS_DATA_PATH)

# Dividir los datos en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1) # 10% de los datos para prueba*

# Para el log board de tensorflow
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Crear el modelo
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(70,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(senas.shape[0], activation='softmax'))

# Compilar el modelo
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Entrenar el modelo
model.fit(x_train, y_train, epochs=100, callbacks=[tb_callback])

# Guardar el modelo
model.save('action.h5')