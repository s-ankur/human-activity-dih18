from keras.layers import Dense, Dropout, Flatten, Input, Activation, Conv2D, BatchNormalization, MaxPooling2D
from keras.models import Model


def cnn2d_model(input_shape, num_classes):
    x = Input(input_shape)
    x = Conv2D(32, 3, 3, border_mode='same', input_shape=input_shape)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, 3, 3)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

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

    model = Model(input_shape, x)
    return model
