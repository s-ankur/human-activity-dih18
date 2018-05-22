from config import *
from model import * 
import sys
import numpy as np
X_test=load_images(sys.argv[1])
y_pred=model.predict(X_test)
y_pred = np.argmax(1)
for pred in y_pred:
    print(actions[pred])
