import sys

from keras.models import model_from_json

from config import *
from dataset import categories
from utility.cv_utils import *

model = model_from_json(os.path.join('results', 'saved.json'))
model.load_weights(os.path.join('results', 'saved.h5py'))

if len(sys.argv) == 2:
    video_path = sys.argv[1]
else:
    video_path = 0
video = Video(video_path)
fourcc = cv2.VideoWriter_fourcc(*"MPEG")
clip = cv2.VideoWriter('demo2.avi', fourcc, 5, (416, 416), True)

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
