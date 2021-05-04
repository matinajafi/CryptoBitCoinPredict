import credentials as cred
from binance.client import Client
import csv
import pandas as pd
import datetime
import parameters as pa
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import numpy as np

############################################

client = Client(cred.API_KEY, cred.API_SECRET)

############################################

custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")


def get_historical_candle_data_binance(pair, period, save_path):
    k_lines = list(client.get_historical_klines(pair, Client.KLINE_INTERVAL_1MINUTE, period))
    k_lines = [[str(datetime.fromtimestamp(float(str(line[0])[:10] + "." + str(line[0])[10:])))] + line[1:6] for
               line in
               k_lines]

    k_lines = [['Date', 'Open', 'High', 'Low', 'Close', 'Volume']] + k_lines
    with open(save_path, 'w', newline='') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_NONE, delimiter=',')
        writer.writerows(k_lines)


def make_dataframe_and_add_y(csv_path=pa.CSV_PATH):
    df = pd.read_csv(csv_path, parse_dates=["Date"], date_parser=custom_date_parser)
    df = df.sort_values('Date')
    df['Y'] = df['Close'][1:]
    df['Y'] = df['Y'].shift(-1)
    df = df[:-1]
    return df


def scale_data_for_lstm(df):
    scaler = MinMaxScaler()
    close_price = df.Close.values.reshape(-1, 1)
    scaled_close = scaler.fit_transform(close_price)
    scaled_close = scaled_close[~np.isnan(scaled_close)]
    scaled_close = scaled_close.reshape(-1, 1)
    return scaled_close, scaler


def generate_sequences(data, sequence_length):
    result = []

    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    return np.array(result)


def create_train_test_sets(clean_raw_data, seq_len, train_percentage):
    data = generate_sequences(clean_raw_data, seq_len)

    num_train = int(train_percentage * data.shape[0])

    X_train = data[:num_train, :-1, :]
    y_train = data[:num_train, -1, :]

    X_test = data[num_train:, :-1, :]
    y_test = data[num_train:, -1, :]

    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    get_historical_candle_data_binance(pa.PAIR, pa.INITIAL_ANALYSIS_PERIOD, pa.CSV_PATH)

    # with open('BTCUSDT_TRADE_ORDERS.csv', 'w', newline='') as file:
    #     writer = csv.writer(file, quoting=csv.QUOTE_ALL, delimiter=';')
    #     while True:
    #         depth = client.get_order_book(symbol='BTCUSDT')
    #         print("number of bids:", len(depth["bids"]))
    #         print("number of asks:", len(depth["asks"]))
    #         dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    #         writer.writerow([dt_string, *depth["bids"]])
    #         writer.writerow([dt_string, *depth["asks"]])
    #         print("done..")
    #         time.sleep(60)
