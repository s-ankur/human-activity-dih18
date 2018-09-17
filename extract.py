import sys
from random import random

from keras.utils import generic_utils
from sklearn.model_selection import train_test_split
from config import *
from utility.cv_utils import *

try:
    category_names = os.listdir(VIDEO_PATH)
    print("Processing %d categories" % len(category_names))
    os.mkdir(IMAGE_PATH)
    for category_name in category_names:
        print("Processing Category :", category_name)
        source_directory = os.path.join(VIDEO_PATH, category_name)
        all_video_paths = listdirp(source_directory)
        train_video_paths, test_video_paths = train_test_split(all_video_paths, test_size=TEST_TRAIN_SPLIT)
        for train_or_test, video_paths in ('train', train_video_paths), ('test', test_video_paths):
            print('Extracting frames from %d %s videos' % (len(video_paths), train_or_test))
            destination_directory = os.path.join(IMAGE_PATH, train_or_test, category_name)
            os.makedirs(destination_directory, exist_ok=True)
            image_id = 0
            for video_path in video_paths:
                video = Video(video_path)
                progressbar = generic_utils.Progbar(len(video))
                frame_id = 0
                for frame in video:
                    progressbar.add(1)
                    if CHANCE > random():
                        image_path = os.path.join(destination_directory, str(image_id) + '.jpg')
                        imwrite(image_path, frame)
                        image_id += 1
                        frame_id += 1
                sys.stdout.write('\b' * 100)
                sys.stdout.write('Extracted %d frames from %s' % (frame_id, video_path))
                print()
except KeyboardInterrupt:
    print("Interrupted. Exiting.")
