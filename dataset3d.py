from keras.utils import np_utils

from config3d import *
from utility.cv_utils import *


def load_data(categories):
    if len(categories) in (0, 1):
        raise ValueError("Cannot classify %d class" % len(categories))
    data = []
    labels = []
    for label, category in enumerate(categories):
        files = glob.glob(os.path.join(category, '*'))
        print("Category %-50s  %-7d files" % (category, len(files)))
        for file in files:
            video = Video(file)
            frame_array = []
            for frame in video:
                if CHANNELS == 1:
                    frame = im2gray(frame)
                frame_array.append(frame)
            frame_array = np.array(frame_array)
            data.append(frame_array)
            labels.append(label)
    X = np.array(data).transpose((0, 2, 3, 4, 1))
    X = X.reshape((X.shape[0], *SIZE3D, DEPTH, CHANNELS))
    y = np.array(labels)
    y = np_utils.to_categorical(y, len(categories))
    print('X.shape:', X.shape)
    print('y.shape:', y.shape)
    return X, y
