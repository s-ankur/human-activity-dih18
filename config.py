"""
Configuration File For 2D model of this project.
All common parameters of the models can be found here
Model Specific parameters found in respective files of the models.
Available Models:
cnn2d
cnn2d_small
dummy
"""

VIDEO_PATH = 'videos'
IMAGE_PATH = 'images'
RESULT_PATH = 'results'
MODEL_NAME = 'saved'
DATASET='ucf101'
############### CONFIG_CNN ##############
MODEL = 'cnn2d'
EXTRACT = False
CHANCE = .05
TEST_TRAIN_SPLIT = .2
SIZE = (64, 64)
CHANNELS = 3
BATCH_SIZE = 128
EPOCHS = 30
############### CONFIG ##############
