from model import *
from dataset import *
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from history import *
from keras.callbacks import CSVLogger


def fit(X, y):
    if not os.path.isdir(RESULT_PATH):
        os.mkdir(RESULT_PATH)
    csv_logger = CSVLogger(os.path.join(RESULT_PATH, 'result.csv'), append=True, separator=';')
    y = to_categorical(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_TRAIN_SPLIT, random_state=42)
    data_generator = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=.1,
        horizontal_flip=True)

    history = model.fit_generator(data_generator.flow(X_train, y_train, BATCH_SIZE),
                                  steps_per_epoch=len(X_train) / BATCH_SIZE,
                                  epochs=EPOCHS,
                                  verbose=True,
                                  callbacks=[csv_logger],
                                  validation_data=(X_test, y_test))

    plot_history(history.history, RESULT_PATH)


if __name__ == "__main__":
    try:
        X, y = load_data(categories)
        print('X.shape:', X.shape)
        print('y.shape:', y.shape)
        fit(X, y)
    finally:
        save_model()
