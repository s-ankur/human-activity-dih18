############### CONFIG ##############

VIDEO_PATH = 'videos'
IMAGE_PATH = 'images'
RESULT_PATH = 'results'
MODEL_NAME = 'saved.h5'
CHANCE = .1
SIZE = (48, 48)
SIZE3D = (200, 200)
MODE = 'color'
BATCH_SIZE = 50
EPOCHS = 20
BACKGROUND_SUBTRACTION = False
TEST_TRAIN_SPLIT = .33

############### CONFIG ##############


import os

for path in (IMAGE_PATH, RESULT_PATH):
    if not os.path.isdir(path):
        print("Creating directory ", path)
        os.mkdir(path)
    else:
        raise Warning("Directory already exists, Won't overwrite", path)
