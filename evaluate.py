from keras.callbacks import CSVLogger, TensorBoard
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
import sklearn.metrics


def plot_history(history, result_path):
    for metric in history:
        plt.plot(history[metric], marker='.', label=metric)
        plt.title('Model Metrics')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='center right')
    plt.savefig(os.path.join(result_path, 'metrics.png'))
    plt.close()


def save_metrics(y_test, y_pred, result_path):
    y_test = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_pred, axis=1)

    with open('classification_report', 'w') as file:
        file.write(sklearn.metrics.classification_report(y_test, y_pred, digits=5))

    confusion_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred)
    confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    np.savetxt(os.path.join(result_path, "confusion_matrix.csv"), confusion_matrix, delimiter=",", fmt='%.4f')

    plt.imshow(confusion_matrix, interpolation='nearest', cmap='hot')
    plt.colorbar()

    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i,
                 '%.2f' % confusion_matrix[i, j],
                 horizontalalignment="center",
                 color="white" if confusion_matrix[i, j] > .5 else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(result_path, 'confusion_matrix.png'))
    plt.close()


def logger(result_path):
    csvlogger = CSVLogger(os.path.join(result_path, 'result.csv'), append=True, separator=';')
    tensorboard = TensorBoard(log_dir=os.path.join(result_path, 'logs'), )
    return [tensorboard, csvlogger]
