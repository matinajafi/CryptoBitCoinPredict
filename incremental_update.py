from tensorflow import keras
import data_preparation as dp
import training as tr
import parameters as pa
import time
import pandas as pd


def refresh_csv_files(original_csv_path, update_csv_path):
    original = open(original_csv_path, "a")
    update = open(update_csv_path, "r")
    flag = True
    for line in update:
        if flag:
            flag = False
            continue
        original.write(line)
    original.close()
    update.close()
    open(update_csv_path, 'w').close()


if __name__ == '__main__':
    model = keras.models.load_model('models/updated_model')
    train_df = dp.make_dataframe_and_add_y()
    while True:
        dp.get_historical_candle_data_binance(pa.PAIR, pa.UPDATE_PERIOD, pa.UPDATES_CSV)
        test_df = dp.make_dataframe_and_add_y(pa.UPDATES_CSV)
        print("new data fetched!")
        print(test_df.head())
        print(test_df.shape)

        scaled_test_df, test_scaler = dp.scale_data_for_lstm(test_df)
        scaled_train_df, train_scaler = dp.scale_data_for_lstm(train_df)

        train_data = dp.generate_sequences(scaled_train_df, pa.SEQ_LEN)
        test_data = dp.generate_sequences(scaled_test_df, pa.SEQ_LEN)

        X_train = train_data[:, :-1, :]
        y_train = train_data[:, -1, :]

        X_test = test_data[:, :-1, :]
        y_test = test_data[:, -1, :]

        tr.update_model(model, X_train, y_train)

        enter = input("press enter to continue...")
        tr.test(model, test_scaler, X_test, y_test)

        refresh_csv_files(pa.CSV_PATH, pa.UPDATES_CSV)
        train_df = train_df.append(test_df)
        print("merged!")
        print(train_df)

        time.sleep(300)
