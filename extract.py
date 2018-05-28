from utility.cv_utils import *
from config import *
import os,glob
from random import random


category_names=glob.glob(os.path.join(VIDEO_PATH,'*'))
try:
    os.mkdir(IMAGE_PATH)
except:pass

for category_name in category_names:
    try:
        os.mkdir(category_name.replace(VIDEO_PATH,IMAGE_PATH))
    except:
        pass
    video_names=glob.glob(os.path.join(category_name,'*'))
    for video_name in video_names:
        video=Video(video_name)
        image_id=0
        try:
            for frame in video:
                if CHANCE>random():
                    image_name=os.path.join(category_name.replace(VIDEO_PATH,IMAGE_PATH),str(image_id)+'.jpg')
                    print(image_name)
                    imwrite(image_name,frame)
                    image_id+=1
        except:
            pass
                    
