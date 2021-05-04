from tensorflow import keras
import data_preparation as dp
import training as tr

if __name__ == '__main__':
    model = keras.models.load_model('models/updated_model')
    df = dp.make_dataframe_and_add_y()
    print(df.head())
    print(df.shape)
    print("data frame is ready!")

    scaled_df, scaler = dp.scale_data_for_lstm(df)

    X_train, y_train, X_test, y_test = dp.create_train_test_sets(scaled_df, 100, 0.8)
    tr.test(model, scaler, X_test, y_test)
