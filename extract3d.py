from utility.cv_utils import *
from config import *
import os
from random import random


fourcc = cv2.VideoWriter_fourcc("MJPG")

def extract3d():
    category_names=os.listdir(VIDEO_PATH)
    if not os.path.isdir(CLIP_PATH)
        os.mkdir(CLIP_PATH)
    for category_name in category_names:
        source_directory = os.path.join(VIDEO_PATH,category_name)
        destination_directory = os.path.join(CLIP_PATH,category_name)
        video_paths=listdirp(source_directory)
        if not os.path.isdir(destination_directory):
            os.mkdir(destination_directory)
        clip_id=0
        video_names=listdirp(source_directory)
        for video_path in video_paths:
            video=Video(video_path)
                while True:
                    ret,frame=video.input_video.read()
                    if not ret:
                        break
                    if CHANCE>random():
                        clip_name=os.path.join(category_name.replace(VIDEO_PATH,CLIP_PATH),str(clip_id)+'.mp4')
                        print(clip_name)
                        clip=cv2.VideoWriter(clip_name, fourcc, 60,SIZE, True)
                        clip_id+=1
                        for i in range(DEPTH):
                            writer.write(image_name,frame)
                        


