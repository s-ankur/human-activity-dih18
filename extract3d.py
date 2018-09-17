import random

from config3d import *
from utility.cv_utils import *
from sklearn.model_selection import train_test_split

try:
    fourcc = cv2.VideoWriter_fourcc(*"MPEG")
    category_names = os.listdir(VIDEO_PATH)
    print("Processing %d categories" % len(category_names))
    os.mkdir(CLIP_PATH)
    for category_name in category_names:
        print("Processing Category :", category_name)
        source_directory = os.path.join(VIDEO_PATH, category_name)
        all_video_paths = listdirp(source_directory)
        train_video_paths, test_video_paths = train_test_split(all_video_paths, test_size=TEST_TRAIN_SPLIT)
        for train_or_test, video_paths in ('train', train_video_paths), ('test', test_video_paths):
            print('Extracting videos from %d %s videos' % (len(video_paths), train_or_test))
            destination_directory = os.path.join(CLIP_PATH, train_or_test, category_name)
            os.makedirs(destination_directory, exist_ok=True)
            clip_id = 0
            video_names = listdirp(source_directory)
            for video_path in video_paths:
                print(video_path)
                video = Video(video_path)
                while True:
                    ret, frame = video.input_video.read()
                    if not ret:
                        break
                    frame = cv2.resize(frame, SIZE3D)
                    if CHANCE > random.random():
                        clip_name = os.path.join(CLIP_PATH, train_or_test, category_name, str(clip_id) + '.avi')
                        clip = cv2.VideoWriter(clip_name, fourcc, 5, SIZE3D, True)
                        clip_id += 1
                        delete = False
                        for i in range(DEPTH + 1):
                            ret, frame = video.input_video.read()
                            if ret:
                                frame = cv2.resize(frame, SIZE3D)
                                clip.write(frame)
                            else:
                                delete = True
                        clip.release()
                        if delete:
                            os.unlink(clip_name)
                            print("Deleting incomplete clip", clip_name)
except KeyboardInterrupt:
    print("Interrupted. Exiting.")
