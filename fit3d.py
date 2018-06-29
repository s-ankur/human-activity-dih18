from config import *
from model3d import *
from dataset3d import *
from history import *
from sklearn.model_selection import train_test_split
from keras.callbacks import CSVLogger
from keras.utils import np_utils


def fit3d(X, y):
    if not os.path.isdir(RESULT_PATH):
        os.mkdir(RESULT_PATH)
    csv_logger = CSVLogger(os.path.join(RESULT_PATH, 'result.csv'), append=True, separator=';')
    y = np_utils.to_categorical(y, len(categories))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_TRAIN_SPLIT, random_state=42)
    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=EPOCHS,
                        callbacks=[csv_logger],
                        verbose=True)
    plot_history(history.history, RESULT_PATH)


if __name__ == "__main__":
    try:
        X, y = load_data3d(categories)
        print('X.shape:', X.shape)
        print('y.shape:', y.shape)
        fit3d(X, y)
    finally:
        save_model()
