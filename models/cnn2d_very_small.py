from keras.layers import Activation, Conv2D, BatchNormalization,MaxPooling2D
from keras.layers import Input, Dense, Flatten
from keras.models import Model,Sequential
from keras.regularizers import l2


def cnn2d_very_small_model(input_shape, num_classes, l2_regularization=0.1):
    model = Sequential()
    regularization = l2(1)

    model.add(Conv2D(32, (3, 3,), padding='same', input_shape=input_shape,kernel_regularizer=regularization,))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model
