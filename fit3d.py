from config import *
from model3d import *
from dataset3d import *
from sklearn.model_selection import train_test_split
from keras.utils import np_utils


def fit3d(X, y):
    y = np_utils.to_categorical(y, len(categories))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    history=model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=EPOCHS, verbose=1)
    return history
    
def plot_history(history, result_dir):
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_accuracy.png'))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.close()


def save_history(history, result_dir):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'result.txt'), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i])) 


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
