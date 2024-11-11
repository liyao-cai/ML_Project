import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


data = pd.read_csv("C:/Users/86156/Desktop/ML/data1.csv")
# print(data.info())

# Preprocessing variables and choose variables
features = ['3:Temperature_Comedor_Sensor', '4:Temperature_Habitacion_Sensor', '5:Weather_Temperature',
            '6:CO2_Comedor_Sensor', '7:CO2_Habitacion_Sensor', '8:Humedad_Comedor_Sensor', '9:Humedad_Habitacion_Sensor',
            '10:Lighting_Comedor_Sensor', '11:Lighting_Habitacion_Sensor', '13:Meteo_Exterior_Crepusculo', '14:Meteo_Exterior_Viento',
            '22:Temperature_Exterior_Sensor', '23:Humedad_Exterior_Sensor']
target = '9:Humedad_Habitacion_Sensor'

X = data[features].values
y = data[target].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

def create_sequences(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i+time_steps])
        ys.append(y[i+time_steps])
    return np.array(Xs), np.array(ys)

# Create time series data, assuming that the data of the past 8 time steps are used to predict the future
time_steps = 6
X_train_seq, y_train_seq = create_sequences(X_train, y_train, time_steps)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, time_steps)

print("Train shape:", X_train_seq.shape)
print("Test shape:", X_test_seq.shape)

model = tf.keras.models.Sequential()
model.add(LSTM(units=128, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
model.add(Dropout(0.3))
model.add(LSTM(units=128, return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(units=128, return_sequences=False))
model.add(Dropout(0.4))
model.add(Dense(units=1))

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

history = model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=32, validation_data=(X_test_seq, y_test_seq))

# model.save('hvac_lstm_model.h5')


y_pred = model.predict(X_test_seq)


mse = mean_squared_error(y_test_seq, y_pred)
mae = mean_absolute_error(y_test_seq, y_pred)

print(f"Test MSE: {mse}")
print(f"Test MAE: {mae}")

plt.plot(y_test_seq, label='True')
plt.plot(y_pred, label='Prediction')
plt.legend()
plt.title('Humidity Forecast - Actual vs Forecast')
plt.show()
