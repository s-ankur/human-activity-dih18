import numpy as np
from config import *
from model3d import *
from dataset3d import *
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils


def fit3d(X, y):
    y = np_utils.to_categorical(y, len(categories))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=EPOCHS, verbose=1)


if __name__ == "__main__":
    try:

        X, y = load_data3d(categories)
        X = np.array(X.transpose((0, 2, 3, 4, 1)))
        X = X.reshape((X.shape[0], *SIZE3D, 3, 3))

        print('X.shape:', X.shape)
        print('y.shape:', y.shape)
        fit3d(X, y)
    finally:
        save_model()
