from keras.callbacks import CSVLogger, TensorBoard
import matplotlib.pyplot as plt
import os


def plot_history(history, result_path):
    plt.plot(history['acc'], marker='.')
    plt.plot(history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(result_path, 'model_accuracy.png'))
    plt.close()

    plt.plot(history['loss'], marker='.')
    plt.plot(history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_path, 'model_loss.png'))
    plt.close()


def logger(result_path):
    csvlogger = CSVLogger(os.path.join(result_path, 'result.csv'), append=True, separator=';')
    tensorboard = TensorBoard(log_dir=os.path.join(result_path, 'logs'), )
    return [tensorboard, csvlogger]
