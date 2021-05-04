from tensorflow import keras
from tensorflow.keras.layers import Bidirectional, Dropout, Activation, Dense
from tensorflow.python.keras.layers import LSTM
from parameters import WINDOW_SIZE, DROPOUT, ACTIVATION, LOSS, OPTIMIZER, BATCH_SIZE, EPOCHS
import matplotlib.pyplot as plt


def create_model(train_data_shape):
    model = keras.Sequential()
    model.add(Bidirectional(LSTM(WINDOW_SIZE, return_sequences=True), input_shape=(WINDOW_SIZE, train_data_shape)))
    model.add(Dropout(rate=DROPOUT))
    model.add(Bidirectional(LSTM((WINDOW_SIZE * 2), return_sequences=True)))
    model.add(Dropout(rate=DROPOUT))
    model.add(Bidirectional(LSTM(WINDOW_SIZE, return_sequences=False)))
    model.add(Dense(units=1))
    model.add(Activation(ACTIVATION))

    model.compile(
        loss=LOSS,
        optimizer=OPTIMIZER
    )
    return model


def train_model(model, X_train, y_train):
    trained = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        validation_split=0.1
    )
    model.save("models/trained_model_0")
    plt.plot(trained.history['loss'])
    plt.plot(trained.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    return trained


def update_model(model, X_train, y_train):
    updated = model.fit(
        X_train,
        y_train,
        epochs=1,
        batch_size=BATCH_SIZE,
        shuffle=False,
        validation_split=0.1
    )
    model.save("models/updated_model")
    plt.plot(updated.history['loss'])
    plt.plot(updated.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    return updated


def test(model, scaler, X_test, y_test):
    y_hat = model.predict(X_test)

    y_test_inverse = scaler.inverse_transform(y_test)
    y_hat_inverse = scaler.inverse_transform(y_hat)

    plt.plot(y_test_inverse, label="Actual Price", color='green')
    plt.plot(y_hat_inverse, label="Predicted Price", color='red')

    plt.title('Bitcoin price prediction')
    plt.xlabel('Time [Minutes]')
    plt.ylabel('Price')
    plt.legend(loc='best')

    plt.show()
