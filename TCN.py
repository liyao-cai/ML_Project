import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tcn import TCN
import matplotlib.pyplot as plt

data = pd.read_csv("C:/Users/86156/Desktop/ML/data1.csv")
# print(data.info())

# Preprocessing variables and choose variables
features = ['3:Temperature_Comedor_Sensor', '4:Temperature_Habitacion_Sensor', '5:Weather_Temperature',
            '6:CO2_Comedor_Sensor', '7:CO2_Habitacion_Sensor', '8:Humedad_Comedor_Sensor', '9:Humedad_Habitacion_Sensor',
            '10:Lighting_Comedor_Sensor', '11:Lighting_Habitacion_Sensor', '13:Meteo_Exterior_Crepusculo', '14:Meteo_Exterior_Viento',
            '15:Meteo_Exterior_Sol_Oest', '16:Meteo_Exterior_Sol_Est', '17:Meteo_Exterior_Sol_Sud', '18:Meteo_Exterior_Piranometro',
            '22:Temperature_Exterior_Sensor', '23:Humedad_Exterior_Sensor']
target = ['7:CO2_Habitacion_Sensor', '9:Humedad_Habitacion_Sensor', '4:Temperature_Habitacion_Sensor']

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
time_steps = 8
X_train_seq, y_train_seq = create_sequences(X_train, y_train, time_steps)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, time_steps)

print("Train shape:", X_train_seq.shape)
print("Test shape:", X_test_seq.shape)

model = tf.keras.models.Sequential()

model.add(TCN(nb_filters=128,  # number of convolution kernels
              kernel_size=5,  # Convolution kernel size
              dilations=[1, 2, 4, 8, 16, 32],  # Dilated convolution factor
              nb_stacks=2,  # Number of stacking layers
              input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]),  # (time_steps, num_features)
              padding='causal',  # Causal Convolution for Time Series Prediction
              return_sequences=False))


model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(units=3))


model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error')
history = model.fit(X_train_seq, y_train_seq, epochs=250, batch_size=16, validation_data=(X_test_seq, y_test_seq))

y_pred = model.predict(X_test_seq).flatten()

# Calculate the evaluation index for each target feature separately
for i, target in enumerate(target):
    mse = mean_squared_error(y_test_seq[:, i], y_pred[:, i])
    mae = mean_absolute_error(y_test_seq[:, i], y_pred[:, i])
    r2 = r2_score(y_test_seq[:, i], y_pred[:, i])
    print(f"--- Target: {target} ---")
    print(f"Test MSE: {mse}")
    print(f"Test MAE: {mae}")
    print(f"Test R²: {r2}")

# Visualizing prediction results
plt.figure(figsize=(15, 5))
for i, target in enumerate(target):
    plt.subplot(1, 3, i + 1)
    plt.plot(y_test_seq[:, i], label='True')
    plt.plot(y_pred[:, i], label='Prediction')
    plt.title(f'Actual vs Prediction\n{target}')
    plt.legend()
plt.tight_layout()
plt.show()
