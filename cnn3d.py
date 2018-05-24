from config import *
from keras.layers import Activation, Conv3D,Input,Dense, Dropout, Flatten,MaxPooling3D
from keras.layers import BatchNormalization
from keras.models import Model,Sequential

def cnn3d_model(input_shape,num_classes):
    model = Sequential()
    model.add(Conv3D(32, kernel_size=(3, 3, 3), input_shape=input_shape, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(32, kernel_size=(3, 3, 3), border_mode='same'))
    model.add(Activation('softmax'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))
    model.add(Dropout(0.25))

    model.add(Conv3D(64, kernel_size=(3, 3, 3), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), border_mode='same'))
    model.add(Activation('softmax'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model
