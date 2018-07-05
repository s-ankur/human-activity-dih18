from config3d import *
import glob
import os

categories = glob.glob(os.path.join(CLIP_PATH, '*'))
cnn3d_model = __import__(MODEL, fromlist=['cnn3d_model'])
model = cnn3d_model(input_shape=SIZE3D + (DEPTH, 3), num_classes=len(categories))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


def load_model():
    model.load_weights(os.path.join(RESULT_PATH, MODEL_NAME))
    print('Loaded model successfully')


def save_model():
    model.save_weights(os.path.join(RESULT_PATH, MODEL_NAME))
    print('Saved model successfully')


try:
    load_model()
except:
    pass
