import argparse

from model3d import *
from utility.cv_utils import *

parser = argparse.ArgumentParser()
parser.add_argument('video_path', nargs='?', default=0)
parser.add_argument('--show', action='store_true')
args = parser.parse_args()

video = Video(args.video_path)

frame_array = []
for frame in video:
    if CHANNELS == 1:
        frame = im2gray(frame).reshape(*frame.shape[:-1], 1)
    cv2.imshow('window', frame)
    cv2.waitKey(1)
    frame = cv2.resize(frame, SIZE3D)
    frame_array.append(frame)
    if len(frame_array) == 3:
        X_predict = np.array(frame_array).reshape(1, *SIZE3D, CHANNELS, DEPTH)
        X_predict = X_predict.transpose((0, 2, 3, 4, 1))
        X_predict = X_predict.reshape((X_predict.shape[0], *SIZE3D, DEPTH, CHANNELS))
        prediction = model.predict(X_predict)
        index = np.argmax(prediction)
        print(categories[index])
        frame_array.clear()
destroy_window('window')
