import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tcn import TCN
import matplotlib.pyplot as plt

data = pd.read_csv("C:/Users/86156/Desktop/ML/data/combined_csv.csv")
# print(data.info())

# Preprocessing variables and choose variables
features = ['3:Temperature_Comedor_Sensor', '4:Temperature_Habitacion_Sensor', '5:Weather_Temperature',
            '6:CO2_Comedor_Sensor', '7:CO2_Habitacion_Sensor', '8:Humedad_Comedor_Sensor', '9:Humedad_Habitacion_Sensor',
            '10:Lighting_Comedor_Sensor', '11:Lighting_Habitacion_Sensor', '13:Meteo_Exterior_Crepusculo', '14:Meteo_Exterior_Viento',
            '15:Meteo_Exterior_Sol_Oest', '16:Meteo_Exterior_Sol_Est', '17:Meteo_Exterior_Sol_Sud', '18:Meteo_Exterior_Piranometro',
            '22:Temperature_Exterior_Sensor', '23:Humedad_Exterior_Sensor']
targets = ['7:CO2_Habitacion_Sensor', '9:Humedad_Habitacion_Sensor', '4:Temperature_Habitacion_Sensor']

X = data[features].values
y = data[targets].values

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

class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, targets):
        self.validation_data = validation_data
        self.targets = targets
        self.epoch_mse = []
        self.epoch_mae = []
        self.epoch_r2 = []

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.validation_data
        y_pred = self.model.predict(X_val)

        # Calculate MSE, MAE, and R² for each target
        mse = []
        mae = []
        r2 = []
        for i, target in enumerate(self.targets):
            mse.append(mean_squared_error(y_val[:, i], y_pred[:, i]))
            mae.append(mean_absolute_error(y_val[:, i], y_pred[:, i]))
            r2.append(r2_score(y_val[:, i], y_pred[:, i]))

        # Calculate the average and store it
        self.epoch_mse.append(np.mean(mse))
        self.epoch_mae.append(np.mean(mae))
        self.epoch_r2.append(np.mean(r2))

        print(f"Epoch {epoch + 1}: Avg MSE = {np.mean(mse):.4f}, Avg MAE = {np.mean(mae):.4f}, Avg R² = {np.mean(r2):.4f}")



model = tf.keras.models.Sequential()
model.add(TCN(nb_filters=64,
              kernel_size=9,
              dilations=[1, 2, 4, 8, 16, 32],
              nb_stacks=3,
              input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]),
              padding='causal',
              return_sequences=False))
model.add(Dense(units=224, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=160, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=3))

model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error')
metrics_callback = MetricsCallback(validation_data=(X_test_seq, y_test_seq), targets=targets)

history = model.fit(X_train_seq, y_train_seq, epochs=150, batch_size=16,
                    validation_data=(X_test_seq, y_test_seq),
                    callbacks=[metrics_callback])

y_pred = model.predict(X_test_seq)

# Calculate the evaluation index for each target feature separately
for i, target in enumerate(targets):
    mse = mean_squared_error(y_test_seq[:, i], y_pred[:, i])
    mae = mean_absolute_error(y_test_seq[:, i], y_pred[:, i])
    r2 = r2_score(y_test_seq[:, i], y_pred[:, i])
    print(f"--- Target: {target} ---")
    print(f"Test MSE: {mse}")
    print(f"Test MAE: {mae}")
    print(f"Test R²: {r2}")

# Visualizing prediction results
plt.figure(figsize=(15, 5))
for i, target in enumerate(targets):
    plt.subplot(1, 3, i + 1)
    plt.plot(y_test_seq[:, i], label='True')
    plt.plot(y_pred[:, i], label='Prediction')
    plt.title(f'Actual vs Prediction\n{target}')
    plt.legend()
plt.tight_layout()
plt.show()


epochs = range(1, 151)


plt.figure(figsize=(18, 6), dpi=300)

# MSE
plt.subplot(1, 3, 1)
plt.plot(epochs, metrics_callback.epoch_mse, label="MSE")
plt.title("MSE Trend")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend()
plt.grid(alpha=0.3)

plt.ylim(0, 10)

# MAE
plt.subplot(1, 3, 2)
plt.plot(epochs, metrics_callback.epoch_mae, label="MAE")
plt.title("MAE Trend")
plt.xlabel("Epochs")
plt.ylabel("MAE")
plt.legend()
plt.grid(alpha=0.3)

plt.ylim(0, 10)

# R²
plt.subplot(1, 3, 3)
plt.plot(epochs, metrics_callback.epoch_r2, label="R²")
plt.title("R² Trend")
plt.xlabel("Epochs")
plt.ylabel("R²")
plt.legend()
plt.grid(alpha=0.3)
plt.ylim(-10, 1)


plt.tight_layout()
plt.show()
