import os

from keras.models import model_from_json
from keras.optimizers import SGD

from config import *
from dataset import categories


def load_model():
    try:
        model = model_from_json(MODEL_NAME + '.json')
        model.load_weights(os.path.join(RESULT_PATH, MODEL_NAME))
        print('Loaded model successfully')
    except:
        if not os.path.isdir(RESULT_PATH):
            os.mkdir(RESULT_PATH)
        opt = SGD(lr=0.011, decay=1e-4)
        MyModel = getattr(getattr(__import__('models.' + MODEL), MODEL), MODEL + '_model')
        model = MyModel(input_shape=SIZE + (CHANNELS,), num_classes=len(categories))
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])
    return model


def save_model(model):
    with open(os.path.join(RESULT_PATH, MODEL_NAME) + '.json', 'w')as model_file:
        model_file.write(model.to_json())
    model.save_weights(os.path.join(RESULT_PATH, MODEL_NAME) + '.h5')
    print('Saved model successfully')
