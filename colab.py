import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM
from tensorflow.python.keras.layers import LSTM
from tensorflow.keras.models import Sequential

# %matplotlib inline

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 14, 8

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


def to_sequences(data, seq_len):
    d = []

    for index in range(len(data) - seq_len):
        d.append(data[index: index + seq_len])

    return np.array(d)


def preprocess(data_raw, seq_len, train_split):
    data = to_sequences(data_raw, seq_len)

    num_train = int(train_split * data.shape[0])

    X_train_p = data[:num_train, :-1, :]
    y_train_p = data[:num_train, -1, :]

    X_test_p = data[num_train:, :-1, :]
    y_test_p = data[num_train:, -1, :]

    return X_train_p, y_train_p, X_test_p, y_test_p


csv_path = "Yahoo.csv"
if __name__ == '__main__':
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    df = df.sort_values('Date')
    # print(df.head)
    # print(df.shape)

    # show the plot
    ax = df.plot(x='Date', y='Close');
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price (USD)")
    # plt.show()

    # normalize input
    scaler = MinMaxScaler()
    close_price = df.Close.values.reshape(-1, 1)
    scaled_close = scaler.fit_transform(close_price)
    print(scaled_close.shape)
    scaled_close = scaled_close[~np.isnan(scaled_close)]
    scaled_close = scaled_close.reshape(-1, 1)

    # preprocess
    SEQ_LEN = 100

    X_train, y_train, X_test, y_test = preprocess(scaled_close, SEQ_LEN, train_split=0.95)
    print(X_train.shape)
    print(X_test.shape)

    # train
    DROPOUT = 0.2
    WINDOW_SIZE = SEQ_LEN - 1

    model = keras.Sequential()

    model.add(Bidirectional(LSTM(WINDOW_SIZE, return_sequences=True),
                            input_shape=(WINDOW_SIZE, X_train.shape[-1])))
    model.add(Dropout(rate=DROPOUT))

    model.add(Bidirectional(LSTM((WINDOW_SIZE * 2), return_sequences=True)))
    model.add(Dropout(rate=DROPOUT))

    model.add(Bidirectional(LSTM(WINDOW_SIZE, return_sequences=False)))

    model.add(Dense(units=1))

    model.add(Activation('linear'))

    model.compile(
        loss='mean_squared_error',
        optimizer='adam'
    )

    BATCH_SIZE = 64

    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=BATCH_SIZE,
        shuffle=False,
        validation_split=0.1
    )
