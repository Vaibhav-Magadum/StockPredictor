import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional

# Function to fetch stock data
def fetch_stock_data(stock_name, start_date, end_date):
    data = yf.download(stock_name, start=start_date, end=end_date)
    if data.empty:
        st.error("No data found for the given stock symbol and dates. Please try again.")
        return None
    return data

# Function to preprocess data and train the LSTM model
def preprocess_and_train_model(data):
    # Selecting target variable and features
    data = data.reset_index()  # Ensure 'Date' is accessible as a column
    output = pd.DataFrame(data['Close'])
    features = ['Open', 'High', 'Low', 'Volume']

    # Scaling the data
    scaler = MinMaxScaler()
    feature_transform = scaler.fit_transform(data[features])
    feature_transform = pd.DataFrame(feature_transform, columns=features, index=data.index)

    # Splitting the data into train and test
    train_size = int(len(data) * 0.8)
    X_train, X_test = feature_transform[:train_size], feature_transform[train_size:]
    y_train, y_test = output[:train_size], output[train_size:]

    # Data processing for LSTM
    trainX = np.array(X_train)
    testX = np.array(X_test)
    X_train = trainX.reshape(trainX.shape[0], 1, trainX.shape[1])
    X_test = testX.reshape(testX.shape[0], 1, testX.shape[1])

    # Building the LSTM model with Bidirectional layer and Dropout
    lstm = Sequential()
    lstm.add(Bidirectional(LSTM(64, activation='relu', return_sequences=True, input_shape=(1, trainX.shape[1]))))
    lstm.add(Dropout(0.2))  # Dropout to reduce overfitting
    lstm.add(LSTM(32, activation='relu', return_sequences=False))
    lstm.add(Dense(1))
    lstm.compile(optimizer='adam', loss='mean_squared_error')

    # Training the model
    lstm.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1, shuffle=False)

    # Predicting the stock price
    testPredict = lstm.predict(X_test)

    # Predicting the stock price for the next day
    next_day = feature_transform[-1:].values
    next_day = next_day.reshape(1, 1, next_day.shape[1])
    next_day_prediction = lstm.predict(next_day)
    predicted_price_next_day = next_day_prediction[0, 0]

    return predicted_price_next_day, data['Close'].iloc[-1], testPredict, y_test

# Streamlit App
st.title("Stock Price Prediction")

# User inputs for stock data
st.sidebar.header("Stock Data Inputs")
stock_name = st.sidebar.text_input("Stock Symbol (e.g., AAPL)", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-01-01"))

if st.sidebar.button("Fetch and Predict"):
    with st.spinner("Fetching data and training the model..."):
        # Fetch stock data
        stock_data = fetch_stock_data(stock_name, start_date, end_date)

        if stock_data is not None:
            # Display the fetched data
            st.write("Fetched Stock Data:")
            st.dataframe(stock_data.tail())

            # Train and predict stock prices
            try:
                predicted_price, actual_price_next_day, testPredict, y_test = preprocess_and_train_model(stock_data)

                # Plot actual vs predicted prices
                plt.figure(figsize=(10, 6))
                plt.plot(y_test.values, label='Actual Price', color='blue')
                plt.plot(testPredict, label='Predicted Price', color='red')
                plt.title('Stock Price Prediction')
                plt.xlabel('Time')
                plt.ylabel('Price')
                plt.legend()
                st.pyplot(plt)

                # Display the prediction for the next day
                st.write(f"Predicted Stock Price for the Next Day: ${predicted_price:.2f}")
                st.write(f"Actual Stock Price for the Next Day: ${actual_price_next_day:.2f}")

                # Display the prediction error
                st.write(f"Prediction Error: ${abs(predicted_price - actual_price_next_day):.2f}")

            except Exception as e:
                st.error(f"An error occurred during model training or prediction: {e}")
