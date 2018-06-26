from utility.cv_utils import *
from config3d import *
import os
from random import random

fourcc = cv2.VideoWriter_fourcc(*"MPEG")
category_names = os.listdir(VIDEO_PATH)
if not os.path.isdir(CLIP_PATH):
    os.mkdir(CLIP_PATH)
for category_name in category_names:
    source_directory = os.path.join(VIDEO_PATH, category_name)
    destination_directory = os.path.join(CLIP_PATH, category_name)
    video_paths = listdirp(source_directory)
    if not os.path.isdir(destination_directory):
        os.mkdir(destination_directory)
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
            if CHANCE > random():
                clip_name = 'clips/' + os.path.join(category_name.replace(VIDEO_PATH, CLIP_PATH), str(clip_id) + '.avi')
                clip = cv2.VideoWriter(clip_name, fourcc, 5, SIZE3D, True)
                clip_id += 1
                delete = False
                for i in range(DEPTH):
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
