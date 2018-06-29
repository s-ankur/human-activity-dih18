from config3d import *
from cnn3d import cnn3d_model
import glob
import os

categories = glob.glob(os.path.join(CLIP_PATH, '*'))
model = cnn3d_model(input_shape=SIZE3D + (DEPTH, 3), num_classes=len(categories))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


def load_model():
    model.load_weights(MODEL_NAME)
    print('Loaded model successfully')


def save_model():
    model.save_weights(MODEL_NAME)
    print('Saved model successfully')


try:
    load_model()
except:
    pass
