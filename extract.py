from utility.cv_utils import *
from config import *
import os
from random import random

def extract():
    category_names=os.listdir(VIDEO_PATH)
    if not os.path.isdir(IMAGE_PATH)
        os.mkdir(IMAGE_PATH)
    for category_name in category_names:
        source_directory = os.path.join(VIDEO_PATH,category_name)
        destination_directory = os.path.join(IMAGE_PATH,category_name)
        video_paths=listdirp(source_directory)
        if not os.path.isdir(destination_directory):
            os.mkdir(destination_directory)
        image_id=0
        for video_path in video_paths:
            video=Video(video_path)            
            for frame in video:
                if CHANCE>random():
                    image_path=os.path.join(destination_directory,str(image_id)+'.jpg')
                    print(image_path)
                    imwrite(image_path,frame)
                    image_id+=1


