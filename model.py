import glob
import os

from config import *

categories = glob.glob(os.path.join(IMAGE_PATH, '*'))
MyModel = __import__(MODEL).__get__(MODEL+'_model')
model = MyModel(input_shape=SIZE + (CHANNELS,), num_classes=len(categories))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


def load_model():
    if not os.path.isdir(RESULT_PATH):
        os.mkdir(RESULT_PATH)
    model.load_weights(os.path.join(RESULT_PATH, MODEL_NAME))
    print('Loaded model successfully')


def save_model():
    with open(os.path.join(RESULT_PATH, MODEL_NAME) + '.json', 'w')as model_file:
        model_file.write(model.to_json())
    model.save_weights(os.path.join(RESULT_PATH, MODEL_NAME) + '.h5')
    print('Saved model successfully')


try:
    load_model()
except:
    pass
