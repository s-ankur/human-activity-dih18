from time import time
from dataset3d import load_data
from evaluate import *
from model3d import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    X_train, X_test, y_train, y_test  = load_data(categories)
    start_time = time()
    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=EPOCHS,
                        callbacks=logger(RESULT_PATH),
                        verbose=True)
    time_trained = time() - start_time
    plot_history(history.history, RESULT_PATH)
    y_pred = model.predict(X_test)
    save_metrics(y_test, y_pred, time_trained, categories, RESULT_PATH)
except KeyboardInterrupt:
    print("Interrupted Training: Exiting")    
finally:
    save_model()
