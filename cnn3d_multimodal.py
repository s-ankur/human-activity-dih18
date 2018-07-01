from keras.layers import Activation, Conv3D, Dense, Dropout, Flatten, MaxPooling3D, Lambda, Add, Input
from keras.models import Model
from dtcwt.numpy import Transform2d
import numpy as np


def dtcwt3d_layer(input_shape):
    transform = Transform2d()
    f = lambda X: transform.forward(X, 2).lowpass
    if len(input_shape) == 3:
        f = lambda X: np.array([X[:, :, i] for i in range(input_shape[-1])])
        f = lambda X: f(X).reshape((1, input_shape[0] // 2, input_shape[1] // 2, input_shape[2]))
    else:
        f = lambda X: f(X).reshape((1, input_shape[0] // 2, input_shape[1] // 2))
    return Lambda(f)


def cnn3d_model(input_shape, num_classes):
    inp = Input(input_shape)

    y = dtcwt3d_layer(input_shape)(inp)

    x = Conv3D(32, kernel_size=(3, 3, 3), input_shape=input_shape, border_mode='same')(inp)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(3, 3, 3), border_mode='same')(x)

    x = Add()([x, y])

    x = Conv3D(64, kernel_size=(3, 3, 3), border_mode='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(3, 3, 3), border_mode='same')(x)

    x = Flatten()(x)
    x = Dense(512, activation='sigmoid')(x)
    x = Dropout(0.7)(x)
    x = Dense(num_classes, activation='softmax')(x)

    return Model(inp, x)
