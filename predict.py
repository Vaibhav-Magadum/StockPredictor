import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import matplotlib. pyplot as plt
import matplotlib
from sklearn. preprocessing import MinMaxScaler
from keras.layers import LSTM, Dropout
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib. dates as mandates
from sklearn. preprocessing import MinMaxScaler
from sklearn import linear_model
from keras. models import Sequential
from keras. layers import Dense
import keras. backend as K
from keras. callbacks import EarlyStopping
from keras.optimizers import Adam
from keras. models import load_model
from keras. layers import LSTM
from keras. utils.vis_utils import plot_model


# Read the stock data from CSV
df = pd.read_csv('stock_data.csv')
df.head()

#check for null values
print("Dataframe Shape",df.shape)
print("null values present:",df.isnull().values.any())

#plotting close price
df['Close'].plot()
plt.show()

#selecting the target variable and features
output = pd.DataFrame(df['Close'])
features = ['Open', 'High', 'Low', 'Volume']

#scaling the data
scaler = MinMaxScaler()
feature_transform = scaler.fit_transform(df[features])
feature_transform = pd.DataFrame(feature_transform, columns=features, index=df.index)
feature_transform.head()

#splitting the data into train and test
train_size = int(len(df) * 0.8)
test_size = len(df) - train_size

timesplit = TimeSeriesSplit(n_splits=10)
for train_index, test_index in timesplit.split(feature_transform):
    X_train, X_test = feature_transform[:train_size], feature_transform[train_size:]
    y_train, y_test = output[:train_size], output[train_size:]

#data processing for LSTM
trainX = np.array(X_train)
testX = np.array(X_test)
X_train = trainX.reshape(trainX.shape[0], 1, trainX.shape[1])
X_test = testX.reshape(testX.shape[0], 1, testX.shape[1])

#building LSTM model
lstm = Sequential()
lstm.add(LSTM(32,input_shape=(1,trainX.shape[1]),activation='relu',return_sequences=False))
lstm.add(Dense(1))
lstm.compile(optimizer='adam', loss='mean_squared_error')
plot_model(lstm, show_shapes=True, show_layer_names=True)

#training the model
history = lstm.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1, shuffle=False)

#predicting the stock price
testPredict = lstm.predict(X_test)

#plotting the predicted stock price
plt.plot(y_test.values, color='blue', label='Actual Stock Price')
plt.plot(testPredict, color='red', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.show()

# Predicting the stock price for the next day
next_day = feature_transform[-1:].values
next_day = next_day.reshape(1, 1, next_day.shape[1])
next_day_prediction = lstm.predict(next_day)

# Scaling back the predicted value
#next_day_prediction = scaler.inverse_transform(next_day_prediction.reshape(1, -1, 1))

# Extracting the predicted value
predicted_price_next_day = next_day_prediction[0, 0]

# Printing the predicted stock price for the next day
print("Predicted Stock Price for the Next Day:", predicted_price_next_day)

