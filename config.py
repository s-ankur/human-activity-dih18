"""
Configuration File For 2D model of this project.
All common parameters of the models can be found here
Model Specific parameters found in respective files of the models.
Available Models:
cnn2d
cnn2d_small
dummy
"""


RESULT_PATH = 'results'
MODEL_NAME = 'saved'

############### CONFIG_CNN ##############
MODEL = 'cnn2d_small'
EXTRACT = False
CHANCE = .1
TEST_TRAIN_SPLIT = .2
SIZE = (64, 64)
CHANNELS = 1
BATCH_SIZE = 128
EPOCHS = 30
############### CONFIG ##############
