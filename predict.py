import sys

from model import *
from utility.cv_utils import *

if len(sys.argv) == 2:
    video_path = sys.argv[1]
else:
    video_path = glob.glob('/dev/video*')[0]
video = Video(video_path)

for frame in video:
    if CHANNELS == 1:
        frame = im2gray(frame).reshape(*frame.shape[:-1], 1)
    cv2.imshow('window', frame)
    cv2.waitKey(1)
    frame = cv2.resize(frame, SIZE)
    X_predict = frame.reshape((1, *frame.shape))
    prediction = model.predict(X_predict)
    index = np.argmax(prediction)
    print(categories[index])
destroy_window('window')
