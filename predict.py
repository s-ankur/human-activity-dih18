import argparse

from config import *
from dataset import categories
from model import load_model
from utility.cv_utils import *

parser = argparse.ArgumentParser()
parser.add_argument('video_path', nargs='?', default=0)
parser.add_argument('--show', action='store_true')
args = parser.parse_args()

model = load_model()
video = Video(args.video_path)
frame = video.read()
y, x = frame.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*"MPEG")
clip = cv2.VideoWriter('demo2.avi', fourcc, 5, (x, y), True)

for frame in video:
    try:
        inp =  cv2.resize(frame, (416, 416))

        if CHANNELS == 1:
            frame = im2gray(frame).reshape(*frame.shape[:-1], 1)

        frame = cv2.resize(frame, SIZE)
        if CHANNELS == 1:
            X_predict = frame.reshape((1, *frame.shape, 1))
        else:
            X_predict = frame.reshape((1, *frame.shape))
        prediction = model.predict(X_predict)
        index = np.argmax(prediction)
        text = "%s" % (categories[index])
        print(text)
        cv2.putText(inp, text, (10, 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    .4,
                    (255, 0, 0), 2)
        clip.write(inp)
        if args.show:
            cv2.imshow('window', inp)
            cv2.waitKey(1)
    except KeyboardInterrupt:
        print("Exiting")
        break
    except Exception as e:
        raise
if args.show:
    destroy_window('window')
