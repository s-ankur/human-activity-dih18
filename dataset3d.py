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
    X_train = np.array(training_data)
    y_train = np.array(training_labels)
    return X_train, y_train
