from keras.layers import Activation, Conv2D, BatchNormalization
from keras.layers import Input, Dense, Flatten
from keras.models import Model
from keras.regularizers import l2


def cnn2d_small_model(input_shape, num_classes, l2_regularization=0.1):
    regularization = l2(l2_regularization)
    img_input = Input(input_shape)

    x = Conv2D(8, (3, 3), strides=(1, 1),
               kernel_regularizer=regularization,
               use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(8, (3, 3), strides=(1, 1),
               kernel_regularizer=regularization,
               use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    output = Dense(num_classes)(residual)
    output = Activation('softmax', name='predictions')(output)

    model = Model(img_input, output)
    return model
