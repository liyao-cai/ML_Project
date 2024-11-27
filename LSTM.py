import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt

from keras_tuner import HyperModel
from keras_tuner.tuners import RandomSearch

import time
from gc import callbacks


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
time_steps = 4
X_train_seq, y_train_seq = create_sequences(X_train, y_train, time_steps)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, time_steps)

print("Train shape:", X_train_seq.shape)
print("Test shape:", X_test_seq.shape)

def evaluate(model, X_test_seq, y_test_seq):
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


class LSTMNet():
  def __init__(self, num_epochs=10, hidden_layers=[256, 256, 256, 256, 256, 256], dropout_rate=0.4):
    self._num_epochs = num_epochs
    self.learning_rate = 1e-4
    self.hidden_layers = hidden_layers
    self._model = tf.keras.models.Sequential()
    self._model.add(tf.keras.Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
    for hidden_layer in hidden_layers[:-1]:
      self._model.add(LSTM(units=hidden_layer, return_sequences=True))
      self._model.add(Dropout(dropout_rate))
    self._model.add(LSTM(units=hidden_layers[-1], return_sequences=False))
    self._model.add(Dropout(dropout_rate))
    self._model.add(Dense(units=3))

    self._model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mean_squared_error')

  def fit(self, X_train_seq, y_train_seq) ->  tf.keras.callbacks.History:
    return self._model.fit(X_train_seq, y_train_seq, epochs=self._num_epochs, batch_size=16, validation_split=0.1)

  def predict(self, X_test_seq):
    return self._model.predict(X_test_seq)

default_lstm = LSTMNet(num_epochs=100)
start_time = time.time()
default_lstm.fit(X_train_seq, y_train_seq)
training_time_lstm = time.time() - start_time

def tuning_model(hp):
  model = tf.keras.models.Sequential()
  model.add(tf.keras.Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2])))

  num_hidden_layers = hp.Int('num_layers', 1, 6)  # tune hidden layers
  for i in range(num_hidden_layers-1):
    model.add(LSTM(units=hp.Int(f'units_{i}', min_value=64, max_value=256, step=32),
                   activation=hp.Choice('activation', ['relu', 'tanh']),
                   return_sequences=True))
    model.add(Dropout(hp.Float('dropout_rate_{i}', min_value=0.1, max_value=0.6, step=0.1)))
  i = num_hidden_layers-1
  model.add(LSTM(units=hp.Int(f'units_{i}', min_value=64, max_value=256, step=32),
                   activation=hp.Choice('activation', ['relu', 'tanh']),
                   return_sequences=False))
  model.add(Dropout(hp.Float('dropout_rate_{i}', min_value=0.1, max_value=0.6, step=0.1)))
  model.add(Dense(units=3))

  optimizer = hp.Choice('optimizer', ['adam', 'sgd'])  # Optimiser. Adam and SGD.

  learning_rate = hp.Float('learning_rate', min_value=5e-5, max_value=5e-3, sampling='log')  # Learning rate.

  if optimizer == 'adam':
      optimizer = Adam(learning_rate=learning_rate)
  else:
      optimizer = SGD(learning_rate=learning_rate)
  model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])
  return model

total_time = 1*3600  # training for 1 hour
per_trial = 3
n_iter = int(total_time / per_trial / training_time_lstm)
tuner = RandomSearch(
    tuning_model,
    objective='val_mean_squared_error',
    max_trials=n_iter,
    executions_per_trial=per_trial,
    directory='project_keras_tuner',
    project_name="tune lstm")

start_time = time.time()
tuner.search(X_train_seq, y_train_seq, epochs=100, batch_size=16, validation_split=0.1)
tuning_time_lstm = time.time() - start_time
best_lstm_model = tuner.get_best_models()[0]

best_lstm_model.save('best_lstm_model.keras')
