import pandas as pd
import numpy as np
import tensorflow
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

january_dates = pd.date_range(start='2025-01-01', end='2025-01-31', freq='D')

def load_and_preprocess_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[['Date', 'Open', 'Close']]

    scaler = MinMaxScaler(feature_range=(0,1))
    df_scaled = scaler.fit_transform(df[['Open', 'Close']])

    def create_dataset(data, time_step=60):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 1])
        return np.array(X), np.array(y)
    
    time_step = 60
    X, y = create_dataset(df_scaled, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    return train_test_split(X, y, test_size=0.2, shuffle=False), scaler


def build_lstm_model(X_train):
    model = tensorflow.keras.Sequential()
    model.add(tensorflow.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(tensorflow.keras.layers.LSTM(units=50, return_sequences=False))
    model.add(tensorflow.keras.layers.Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(X_train, y_train, model):
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    return model

def predict_stock_prices(model, X_test, scaler, january_dates):

    predicted_prices = model.predict(X_test)

    predicted_prices = predicted_prices.reshape(-1, 1)

    open_prices = X_test[:, -1, 0].reshape(-1, 1)

    predictions_combined = np.hstack((open_prices, predicted_prices))

    predicted_prices_original_scale = scaler.inverse_transform(predictions_combined)

    january_predictions = predicted_prices_original_scale[-len(january_dates):, 1]

    return january_predictions

def visualize_predictions(predictions, january_dates):
    plt.figure(figsize=(10,6))
    plt.plot(january_dates, predictions, label='Predicted Close Price', color='orange')
    plt.title('Predicted Stock Closing Price for January 2025')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()