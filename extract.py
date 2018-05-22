from utility.cv_utils import *
from config import *
import os,glob
from random import random


actions=glob.glob(os.path.join(VIDEO_PATH,'*'))
try:
    os.mkdir(IMAGE_PATH)
except:pass

for action in actions:
    os.mkdir(action.replace(VIDEO_PATH,IMAGE_PATH))
    video_names=glob.glob('*')
    for video_name in video_names:
        video=Video(video_name)
        for frame in video:
            if CHANCE>random():
                imwrite(action.replace(VIDEO_PATH,IMAGE_PATH),frame)
                
