from keras.layers import Activation, Conv3D, Dense, Dropout, Flatten, MaxPooling3D, Add, Input
from keras.models import Model

from transform import dtcwt3d_layer


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
