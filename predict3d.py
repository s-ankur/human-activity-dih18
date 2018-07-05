import sys
from model3d import *
from utility.cv_utils import *

if len(sys.argv) == 2:
    video_path = sys.argv[1]
else:
    video_path = glob.glob('/dev/video*')[0]
video = Video(video_path)

frame_array = []
for frame in video:
    cv2.imshow('window', frame)
    cv2.waitKey(1)
    frame = cv2.resize(frame, SIZE3D)
    frame_array.append(frame)
    if len(frame_array) == 3:
        X_predict = np.array(frame_array).reshape(1, *SIZE3D, 3, DEPTH)
        X_predict = X_predict.transpose((0, 2, 3, 4, 1))
        X_predict = X_predict.reshape((X_predict.shape[0], *SIZE3D, DEPTH, 3))
        prediction = model.predict(X_predict)
        index = np.argmax(prediction)
        print(categories[index])
        frame_array.clear()
destroy_window('window')
