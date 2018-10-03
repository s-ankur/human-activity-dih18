import sys

from keras.models import model_from_json

from config import *
from dataset import categories
from utility.cv_utils import *
from model import load_model
import numpy as np
import numpy.random as random
model=load_model()
SHOW=False
if len(sys.argv) == 2:
    video_path = sys.argv[1]
else:
    video_path = 0
video = Video(video_path)
frame = video.read()
y,x=frame.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*"MPEG")
clip = cv2.VideoWriter('demo2.avi', fourcc, 5, (x, y), True)

try:
    for frame in video:
        inp = frame
        if CHANNELS == 1:
            frame = im2gray(frame).reshape(*frame.shape[:-1], 1)

        frame = cv2.resize(frame, SIZE)
        X_predict = frame.reshape((1, *frame.shape,1))
        prediction = model.predict(X_predict)
        index = np.argmax(prediction)
        text ="%s %.3f"%(categories[index],prediction[0][index])
        print(text)
        cv2.putText(inp,text,(10, 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            .4,
                            (255, 255, 255), 2)
        clip.write(inp)
        if SHOW:
            cv2.imshow('window', inp)
            cv2.waitKey(1)
    if SHOW:
        destroy_window('window')
except KeyboardInterrupt:
    print("Exiting")
