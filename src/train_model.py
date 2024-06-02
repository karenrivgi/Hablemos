"""
Módulo para entrenar el modelo de clasificación de señas.

Se entrena un modelo de clasificación de señas utilizando un modelo LSTM.
Se permite probar el modelo comparando las predicciones con las etiquetas reales.
"""

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

# ---------------------------------------------------------------------

senas = np.array(['Hola','Amigo', 'Adios']) # Señas a interpretar
PROCESS_DATA_PATH = os.path.join('MP_Data') # Carpeta con los datos procesados
x, y = load_processed_data(PROCESS_DATA_PATH) # Etiquetar los datos procesados

# Dividir los datos en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1) # 10% de los datos para prueba*

# Para el log board de tensorflow
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# ---------------------------------------------------------------------
# Crear el modelo

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(senas.shape[0], activation='softmax'))

# Compilar el modelo
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Entrenar el modelo
# model.fit(x_train, y_train, epochs=90, callbacks=[tb_callback])

# Guardar el modelo
# model.save('action.h5')

# ---------------------------------------------------------------------
# Probar el modelo
model.load_weights('action.h5')

from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

# Matriz de confusión y accuracy, si el modelo es bueno, 
# la diagonal de la matriz de confusión debería ser alta

yhat = model.predict(x_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()
print(multilabel_confusion_matrix(ytrue, yhat))
print(accuracy_score(ytrue, yhat))

# Probar predicciones (Comentar línea para entrenar y guardar el modelo)
res = model.predict(x_test)
while True:
    prueba = int(input("Ingrese el indice de la prueba: "))
    print("Resultado esperado:")
    print(senas[np.argmax(y_test[prueba])])
    print("Resultado obtenido:")
    print(senas[np.argmax(res[prueba])])


