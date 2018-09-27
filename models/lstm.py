"""
A collection of models we'll use to attempt to classify videos.
"""
from keras.layers import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential


def lstm_model(input_shape, num_classes):
    """Build a simple LSTM network. We pass the extracted features from
    our CNN to this model predomenently."""
    # Model.
    model = Sequential()
    model.add(LSTM(2048, return_sequences=False,
                   input_shape=(3, 2048),
                   dropout=0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model
