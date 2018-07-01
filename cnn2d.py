from keras.layers import Dense, Dropout, Flatten
from keras.layers import Activation, Conv2D, GlobalAveragePooling2D, BatchNormalization, SeparableConv2D, MaxPooling2D
from keras.models import Sequential


def cnn2d_model(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(32, 3, 3, border_mode='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, 3, 3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, 3, 3, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, 3, 3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.6))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model
