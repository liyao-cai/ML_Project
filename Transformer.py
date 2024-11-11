import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, MultiHeadAttention, LayerNormalization, Input, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


data = pd.read_csv("C:/Users/86156/Desktop/ML/data1.csv")


features = ['3:Temperature_Comedor_Sensor', '4:Temperature_Habitacion_Sensor', '5:Weather_Temperature',
            '6:CO2_Comedor_Sensor', '7:CO2_Habitacion_Sensor', '8:Humedad_Comedor_Sensor', '9:Humedad_Habitacion_Sensor',
            '10:Lighting_Comedor_Sensor', '11:Lighting_Habitacion_Sensor', '13:Meteo_Exterior_Crepusculo', '14:Meteo_Exterior_Viento',
            '22:Temperature_Exterior_Sensor', '23:Humedad_Exterior_Sensor']
target = '9:Humedad_Habitacion_Sensor'

X = data[features].values
y = data[target].values


scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into 90% training set and 10% test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

# Creating time series data
def create_sequences(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i+time_steps])
        ys.append(y[i+time_steps])
    return np.array(Xs), np.array(ys)

# Predicting the future using the past 6 time steps
time_steps = 6
X_train_seq, y_train_seq = create_sequences(X_train, y_train, time_steps)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, time_steps)

print("Train shape:", X_train_seq.shape)
print("Test shape:", X_test_seq.shape)


def transformer_model(input_shape, d_model=128, num_heads=4, ff_dim=256, dropout_rate=0.3):
    inputs = Input(shape=input_shape)

    # Step 0: Project the input features to d_model dimensions
    x = Dense(d_model)(inputs)

    # Step 1: Multi-Head Attention
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    attention_output = Dropout(dropout_rate)(attention_output)
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output)

    # Step 2: Feed-Forward Network (FFN)
    ffn_output = Dense(ff_dim, activation="relu")(attention_output)
    ffn_output = Dropout(dropout_rate)(ffn_output)
    ffn_output = Dense(d_model)(ffn_output)
    ffn_output = LayerNormalization(epsilon=1e-6)(ffn_output + attention_output)

    # Step 3: Global Average Pooling to reduce the sequence dimension
    # This reduces the output from [batch_size, time_steps, d_model] to [batch_size, d_model]
    pooled_output = GlobalAveragePooling1D()(ffn_output)

    # Step 4: Output a single value (next time step prediction)
    outputs = Dense(1)(pooled_output)

    # Step 5: Build and compile the model
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    return model


input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])  # (time_steps, num_features)
model = transformer_model(input_shape)


history = model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=32, validation_data=(X_test_seq, y_test_seq))

# Save the model
# model.save('hvac_transformer_model.h5')

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

