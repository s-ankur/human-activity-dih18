from keras.utils import np_utils
from config import *
from utility.cv_utils import *


def load_data(categories):
    if len(categories) in (0, 1):
        raise ValueError("Cannot classify %d class" % len(categories))

    ret_X=[]
    ret_y=[]
    for train_or_test in 'train','test':
        data = []
        labels = []
        for label, category in enumerate(categories):
            category=category.replace('train',train_or_test)
            files = glob.glob(os.path.join(category, '*'))
            print("%3d. Category %-50s  %-7d files" % (label, category, len(files)))
            for file in files:
                image = imread(file)
                image = cv2.resize(image, SIZE)
                if CHANNELS == 1:
                    image = im2gray(image).reshape(*image.shape[:-1], 1)
                data.append(image)
                labels.append(label)
        X = np.array(data)
        X = X/255.
        y = np.array(labels)
        y = np_utils.to_categorical(y, len(categories))

        print('X_%s.shape:'%train_or_test, X.shape)
        print('y_%s.shape:'%train_or_test, y.shape)

        ret_X.append(X)
        ret_y.append(y)
    return ret_X+ret_y
    
