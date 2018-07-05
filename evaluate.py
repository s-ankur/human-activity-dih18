from keras.callbacks import CSVLogger, TensorBoard
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
import sklearn.metrics
import json


def plot_history(history, result_path):
    if history.get('acc') and history.get('val_acc'):
        plt.plot(history['acc'], marker='.')
        plt.plot(history['val_acc'], marker='.')
        plt.title('Model accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.grid()
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(result_path, 'metrics.png'))
        plt.close()

    if history.get('loss') and history.get('val_loss'):
        plt.plot(history['loss'], marker='.')
        plt.plot(history['val_loss'], marker='.')
        plt.title('Model loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.grid()
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(result_path, 'metrics.png'))
        plt.close()


def save_metrics(y_test, y_pred, time_trained, categories, result_path):
    y_test = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)

    with open('metrics.json', 'w') as metrics_file:
        metrics = dict(time_trained=time_trained,
                       accuracy=accuracy)
        json.dump(metrics, metrics_file, sort_keys=True)

    with open('classification_report.txt', 'w') as classification_report_file:
        classification_report = sklearn.metrics.classification_report(y_test, y_pred, target_names=categories, digits=5)
        classification_report_file.write(classification_report)

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
