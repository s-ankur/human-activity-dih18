############### CONFIG ##############

VIDEO_PATH = 'videos'
CLIP_PATH = 'clips'
RESULT_PATH = 'results3d'
MODEL_NAME = 'saved3d.h5'
CHANCE = .01
DEPTH = 3
SIZE3D = (64, 64)
CHANNELS = 3
BATCH_SIZE = 128
EPOCHS = 30
BACKGROUND_SUBTRACTION = False
TEST_TRAIN_SPLIT = .33

############### CONFIG ##############

import os

for variable in globals():
    if variable.endswith('PATH'):
        if not os.path.isdir(variable):
            print("Creating directory ", variable)
            os.mkdir(variable)
        else:
            print("Warning: Directory already exists", variable)