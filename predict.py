from config import *
from model import *
from utility.cv_utils import *
import sys
import numpy as np

video = Video(sys.argv[1])
for frame in video:
    frame = cv2.resize(frame, SIZE)
    cv2.imshow('window', frame)
    cv2.waitKey(1)
    prediction = model.predict(frame.reshape((1, *frame.shape)))
    # print(prediction)
    index = np.argmax(prediction)
    print(categories[index])
destroy_window('window')
