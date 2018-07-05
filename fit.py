from time import time
from sklearn.model_selection import train_test_split
from model import *
from dataset import load_data
from evaluate import *

try:
    X, y = load_data(categories)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_TRAIN_SPLIT, random_state=42)
    start_time = time()
    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=EPOCHS,
                        callbacks=logger(RESULT_PATH),
                        verbose=True)
    time_trained = time() - start_time
    if not os.path.isdir(RESULT_PATH):
        os.mkdir(RESULT_PATH)
    plot_history(history.history, RESULT_PATH)
    y_pred = model.predict(X_test)
    save_metrics(y_test, y_pred, time_trained, RESULT_PATH)
finally:
    save_model()
