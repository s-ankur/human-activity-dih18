"""
############### CONFIG_CNN3D ##############
VIDEO_PATH = 'videos'
CLIP_PATH = 'clips'
RESULT_PATH = 'results3d'
MODEL_NAME = 'saved3d'
MODEL = 'cnn3d'
CHANCE = .01
TEST_TRAIN_SPLIT = .33
SIZE3D = (64, 64)
DEPTH = 3
CHANNELS = 3
BATCH_SIZE = 128
EPOCHS = 30
############### CONFIG ##############
"""


############### CONFIG_LSTM ##############
VIDEO_PATH = 'videos'
CLIP_PATH = 'clips'
RESULT_PATH = 'resultslstm'
MODEL_NAME = 'savedlstm'
MODEL = 'lstm'
EXTRACT=True
CHANCE = .1
TEST_TRAIN_SPLIT = .2
SIZE3D = (64, 64)
CHANNELS = 1
DEPTH = 3
BATCH_SIZE = 128
EPOCHS = 30
############### CONFIG ##############
