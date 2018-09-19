from keras.utils import np_utils
from config3d import *
from utility.cv_utils import *
from extractor import Extractor


def load_data(categories):
    if len(categories) in (0, 1):
        raise ValueError("Cannot classify %d class" % len(categories))

    ret_X = []
    ret_y = []
    for train_or_test in 'train', 'test':
        data = []
        labels = []
        for label, category in enumerate(categories):
            category = category.replace('train', train_or_test)
            files = glob.glob(os.path.join(category, '*'))
            print("%3d. Category %-50s  %-7d files" % (label, category, len(files)))
            for file in files:
                video = Video(file)
                frame_array = []
                for frame in video:
                    if CHANNELS == 1:
                        frame = im2gray(frame).reshape(frame.shape[:-1], 1)
                    frame_array.append(frame)
                frame_array = np.array(frame_array)
                data.append(frame_array)
                labels.append(label)
        if not EXTRACT:
            X = np.array(data).transpose((0, 2, 3, 4, 1))
            X = X.reshape((X.shape[0], *SIZE3D, DEPTH, CHANNELS))
            X /= 255
        else:
            extractor = Extractor()
            X = []
            for frame_array in data:
                frame_array = extractor.extract(frame_array)
                X.append(frame_array)
            X = np.array(X)
        y = np.array(labels)
        y = np_utils.to_categorical(y, len(categories))

        print('X_%s.shape:' % train_or_test, X.shape)
        print('y_%s.shape:' % train_or_test, y.shape)

        ret_X.append(X)
        ret_y.append(y)
    return ret_X + ret_y
