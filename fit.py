from model import *
from dataset import load_data
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from evaluate import *

try:
    X, y = load_data(categories)
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
                                  callbacks=logger(RESULT_PATH),
                                  validation_data=(X_test, y_test))
    y_pred = model.predict(X_test)
    plot_history(history.history, RESULT_PATH)
    save_metrics(y_test, y_pred, RESULT_PATH)
finally:
    save_model()
