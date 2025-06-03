import os
import logging
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
except ImportError:
    st.error("TensorFlow is not installed or incompatible with your Python version.")
    st.stop()

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    st.error("statsmodels is not installed or incompatible with your Python version.")
    st.stop()

# Logging setup
logging.basicConfig(
    filename='forecasting.log',
    filemode='w',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Streamlit Interface
st.title("Electric Production Forecasting App")
st.markdown("Upload your electric production CSV file to generate forecasts using ARIMA and LSTM models.")

uploaded_file = st.file_uploader("Upload Electric_Production.csv", type=['csv'])

if uploaded_file is not None:
    with st.spinner('Loading data...'):
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = ['Date', 'Production']
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df = df.asfreq('MS')
            df = df.ffill()
            logging.info("CSV file loaded and preprocessed.")
        except Exception as e:
            st.error(f"Error loading file: {e}")
            logging.error(f"Error loading file: {e}")
            st.stop()

    st.success("File loaded successfully!")

    # Visualization
    st.subheader("Production Over Time")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df['Production'])
    ax.set_title("Electric Production Over Time")
    st.pyplot(fig)

    # ARIMA Forecasting
    st.subheader("ARIMA Forecast")
    with st.spinner('Training ARIMA model...'):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arima_model = ARIMA(df['Production'], order=(5, 1, 0))
            arima_result = arima_model.fit()
            forecast = arima_result.forecast(steps=24)

        forecast_index = pd.date_range(df.index[-1], periods=25, freq='MS')[1:]
        true_vals = df['Production'][-24:].values
        arima_mse = mean_squared_error(true_vals, forecast[:len(true_vals)])
        arima_rmse = np.sqrt(arima_mse)

    st.write(f"**ARIMA MSE:** {arima_mse:.2f}  |  **RMSE:** {arima_rmse:.2f}")
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    ax2.plot(df.index[-100:], df['Production'][-100:], label='Actual')
    ax2.plot(forecast_index, forecast, label='ARIMA Forecast', linestyle='--')
    ax2.legend()
    ax2.set_title("ARIMA Forecast - Next 24 Months")
    st.pyplot(fig2)

    # LSTM Forecasting
    st.subheader("LSTM Forecast")
    with st.spinner('Training LSTM model...'):
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[['Production']])

        look_back = 12
        X, y = [], []
        for i in range(look_back, len(scaled_data)):
            X.append(scaled_data[i - look_back:i, 0])
            y.append(scaled_data[i, 0])

        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        model = Sequential([
            LSTM(units=100, return_sequences=True, input_shape=(X.shape[1], 1)),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(units=1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=25, batch_size=32, verbose=0)

        predicted = model.predict(X_test)
        predicted = scaler.inverse_transform(predicted.reshape(-1, 1))
        y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

        lstm_mse = mean_squared_error(y_test_scaled, predicted)
        lstm_rmse = np.sqrt(lstm_mse)

    st.write(f"**LSTM MSE:** {lstm_mse:.2f}  |  **RMSE:** {lstm_rmse:.2f}")
    fig3, ax3 = plt.subplots(figsize=(14, 6))
    ax3.plot(df.index[-len(predicted):], y_test_scaled, label='Actual')
    ax3.plot(df.index[-len(predicted):], predicted, label='LSTM Prediction')
    ax3.legend()
    ax3.set_title("LSTM Forecast vs Actual")
    st.pyplot(fig3)

    # Future forecast with LSTM
    st.subheader("12-Month Forecast (LSTM)")
    with st.spinner('Generating future forecast...'):
        last_seq = scaled_data[-look_back:]
        future_preds = []
        current_seq = last_seq.copy()

        for _ in range(12):
            pred = model.predict(current_seq.reshape(1, look_back, 1), verbose=0)[0][0]
            future_preds.append(pred)
            current_seq = np.append(current_seq[1:], [[pred]], axis=0)

        future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
        future_dates = pd.date_range(df.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')

    fig4, ax4 = plt.subplots(figsize=(14, 6))
    ax4.plot(df['Production'], label='History')
    ax4.plot(future_dates, future_preds, label='Future Forecast (LSTM)', linestyle='--')
    ax4.legend()
    ax4.set_title("Electric Production Forecast - Next 12 Months")
    st.pyplot(fig4)

    st.success("Forecasting complete! See plots and metrics above.")
else:
    st.info("Awaiting CSV upload...")


