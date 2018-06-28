from utility.cv_utils import *
from config3d import *


def load_data3d(categories, size=SIZE3D, mode=MODE):
    training_data = []
    training_labels = []
    for label, category in enumerate(categories):
        files = glob.glob(os.path.join(category, '*'))
        print("Category %s --- %d files" % (category, len(files)))
        for file in files:
            video = Video(file)
            frame_array = []
            for frame in video:
                frame_array.append(frame)
            frame_array = np.array(frame_array)
            training_data.append(frame_array)
            training_labels.append(label)
    X_train = np.array(training_data).transpose((0, 2, 3, 4, 1))
    print(X_train.shape)
    X_train = X_train.reshape((X_train.shape[0], *SIZE3D, DEPTH,3))
    y_train = np.array(training_labels)
    return X_train, y_train
