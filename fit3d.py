from config import *
from model3d import *
from dataset3d import *
from history import *
from sklearn.model_selection import train_test_split
from keras.utils import np_utils


def fit3d(X, y):
    y = np_utils.to_categorical(y, len(categories))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    history=model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=EPOCHS, verbose=1)
    return history
    

if __name__ == "__main__":
    try:
        X, y = load_data3d(categories)
        print('X.shape:', X.shape)
        print('y.shape:', y.shape)
        history=fit3d(X, y)
        plot_history(history,RESULT_PATH)
        save_history(history,RESULT_PATH)
    finally:
        save_model()
