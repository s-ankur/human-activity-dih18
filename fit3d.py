from model3d import *
from dataset3d import load_data
from evaluate import *
from sklearn.model_selection import train_test_split

try:
    X, y = load_data(categories)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_TRAIN_SPLIT, random_state=42)
    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=EPOCHS,
                        callbacks=logger(RESULT_PATH),
                        verbose=True)
    y_pred = model.predict(X_test)
    plot_history(history.history, RESULT_PATH)
    save_metrics(y_test, y_pred, RESULT_PATH)
finally:
    save_model()
