import sys
import shutil
from random import random

from keras.utils import generic_utils

from config import *
from utility.cv_utils import *

try:
    category_names = os.listdir(VIDEO_PATH)
    print("Processing %d categories" % len(category_names))
    if os.path.isdir(IMAGE_PATH):
        shutil.rmtree(IMAGE_PATH, ignore_errors=True)
    os.mkdir(IMAGE_PATH)
    for category_name in category_names:
        source_directory = os.path.join(VIDEO_PATH, category_name)
        destination_directory = os.path.join(IMAGE_PATH, category_name)
        video_paths = listdirp(source_directory)
        print('Extracting frames from %d videos' % len(video_paths))
        if not os.path.isdir(destination_directory):
            os.mkdir(destination_directory)
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
