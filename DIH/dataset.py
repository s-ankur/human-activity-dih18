from utility.cv_utils import *
from config import *
import glob
import os

actions=glob.glob(os.path.join(ACTION_PATH,'*'))

def load_data(categories=actions,size=SIZE,mode=MODE):
    training_data = []
    training_labels = []
    for label,category in enumerate(categories):
        files = glob.glob(os.path.join(category,'*'))
        print("Category %s --- %d files"%(category,len(files)))
        for file in files:
            image = imread(file,mode)
            image = cv2.resize(image,size)
            training_data.append(image)  
            training_labels.append(label)
    X_train=np.array(training_data)
    y_train=np.array(training_labels)
    return X_train, y_train

