import numpy as np
from dtcwt.numpy import Transform2d
from keras.layers import Dense, Dropout, Flatten, Input, Activation, Conv2D, BatchNormalization, MaxPooling2D, Lambda, \
    Add
from keras.models import Model


def dtcwt_layer(input_shape):
    transform = Transform2d()
    f = lambda X: transform.forward(X, 2).lowpass
    if len(input_shape) == 3:
        f = lambda X: np.array([X[:, :, i] for i in range(input_shape[-1])])
        f = lambda X: f(X).reshape((1, input_shape[0] // 2, input_shape[1] // 2, input_shape[2]))
    else:
        f = lambda X: f(X).reshape((1, input_shape[0] // 2, input_shape[1] // 2))
    return Lambda(f)


def cnn2d_model(input_shape, num_classes):
    inp = Input(input_shape)
    y = dtcwt_layer(input_shape)(inp)

    x = Conv2D(16, 3, 3, border_mode='same', input_shape=input_shape)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(16, 3, 3)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Add()([x, y])

    x = Conv2D(64, 3, 3, border_mode='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, 3, 3)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(512)(x)
    x = Activation('relu')(x)
    x = Dropout(0.6)(x)
    x = Dense(num_classes)(x)
    x = Activation('softmax')(x)

    model = Model(inp, x)
    return model
