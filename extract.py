from utility.cv_utils import *
from config import *
import os,glob
from random import random


actions=glob.glob(os.path.join(VIDEO_PATH,'*'))
for action in actions:
    video_names=glob.glob('*')
    for video_name in video_names:
        video=Video(video_name)
        for frame in video:
            if CHANCE>random():
                imwrite(action.replace(VIDEO_PATH,IMAGE_PATH),frame)
                
