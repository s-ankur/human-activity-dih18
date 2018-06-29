import sys
import os
from PIL import Image


def iter_frames(im):
    try:
        i = 0
        while 1:
            im.seek(i)
            imframe = im.copy()
            if i == 0:
                palette = imframe.getpalette()
            else:
                imframe.putpalette(palette)
            yield imframe
            i += 1
    except EOFError:
        pass


file_name = sys.argv[1]
im = Image.open(file_name)
folder_name = file_name.rstrip('.gif')
os.mkdir(folder_name)
for i, frame in enumerate(iter_frames(im)):
    destination_name = os.path.join(folder_name, '%d.png' % i)
    frame.save(destination_name, **frame.info)
    print(destination_name)
