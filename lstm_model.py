import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def run_lstm(series, steps=30):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(series.values.reshape(-1,1))

    X, y = [], []
    for i in range(30, len(data_scaled)):
        X.append(data_scaled[i-30:i])
        y.append(data_scaled[i])

    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(50, activation="relu", input_shape=(30,1)),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=5, batch_size=16, verbose=0)

    last_30 = data_scaled[-30:].reshape(1,30,1)
    pred_scaled = model.predict(last_30)
    pred = scaler.inverse_transform(pred_scaled)

    return float(pred[0][0])
