import data_preparation as dp
import parameters as pa
from parameters import SEQ_LEN
import training as tr

if __name__ == '__main__':
    # print("Starting the program...")
    # dp.get_historical_candle_data_binance(pa.PAIR, pa.INITIAL_ANALYSIS_PERIOD, pa.CSV_PATH)
    # print("Data Fetch Done!")

    df = dp.make_dataframe_and_add_y()
    print(df.head())
    print(df.shape)
    print("data frame is ready!")

    scaled_df, scaler = dp.scale_data_for_lstm(df)
    # print(scaled_df.head())
    # print(scaled_df.shape)
    print("data frame is scaled!")

    X_train, y_train, X_test, y_test = dp.create_train_test_sets(scaled_df, SEQ_LEN, 0.98)
    print("x-train:", X_train.shape, "y-train:", y_train.shape, "x-test:", X_test.shape, "y-test:",
          X_test.shape)
    print("data split completed!")

    model = tr.create_model(X_train.shape[-1])
    print("model created!")
    trained = tr.train_model(model, X_train, y_train)

    enter = input("press enter to continue...")
    print("training Done!")
    tr.test(model, scaler, X_test, y_test)
    print("test Done!")
