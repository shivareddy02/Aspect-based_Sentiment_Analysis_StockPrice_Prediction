import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Define custom CSS style with background image
import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

add_bg_from_local('stock_price1.jpg')    

st.header("Netflix Stock Price Prediction App")
input_days = st.slider('Days', 0, 60)

# Load the data
df = pd.read_csv('NFLX.csv', index_col='Date', parse_dates=True)



def predict_stocks(number):
    df = pd.read_csv('NFLX.csv', index_col='Date', parse_dates=True)

    model = tf.keras.models.load_model('best_lstm_model.h5', compile = False)
    model.compile(optimizer='adam', loss='mse')
    # Sort the data by date
    df = df.sort_values('Date')

    # Create a new dataframe with only the 'Close' column
    data = df.filter(['Adj Close'])

    # Convert the dataframe to a numpy array
    dataset = data.values

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    # Define the number of days to predict in the future
    prediction_days = number

    # Create a list of dates for the prediction period
    last_date = df.index[-1]
    dates = pd.date_range(last_date, periods=prediction_days+1, freq='B')[1:]

    # Predict the future prices
    last_60_days = scaled_data[-30:]
    X_predict = []
    X_predict.append(last_60_days)
    X_predict = np.array(X_predict)
    X_predict = np.reshape(X_predict, (X_predict.shape[0], X_predict.shape[1], 1))

    predicted_prices = []

    for i in range(prediction_days):
        predicted_price = model.predict(X_predict,verbose=0)
        predicted_prices.append(predicted_price[0])
        last_60_days = np.append(last_60_days[1:], predicted_price, axis=0)
        X_predict = np.array([last_60_days])
        X_predict = np.reshape(X_predict, (X_predict.shape[0], X_predict.shape[1], 1))

    # Inverse transform the predicted prices to their original scale
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Create a dataframe of the predicted prices and dates
    predictions = pd.DataFrame(predicted_prices, index=dates, columns=['Adj Close'])

    # Plot the original and predicted stock prices
    fig = plt.figure(figsize=(16,8))
    plt.plot(data['Adj Close'])
    plt.plot(predictions['Adj Close'])
    plt.title('Netflix Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Close Price')

  # Predict the future prices
    plt.legend(['Actual', 'Predicted'])
    st.pyplot(fig)

    return predictions


# To display the current trend of the netflix stock prices
if st.checkbox('Display current value of stock prices'):
    st.dataframe(df['Adj Close'].tail(5))


# To predict the future stock prices according to the input given
if st.button('Make Prediction'):
   prediction = predict_stocks(input_days)
   st.write("Predicted Value", prediction)
