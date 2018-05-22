from __future__ import print_function, division
import math,cmath

def dist(a, b=[0,0]):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def closest(points, source,key=lambda x:x):
    points.sort(key=lambda x: dist(key(x), source))
    return points.pop(0)

def angle(v1,v2=(1,0)):
    v1=complex(*v1)
    v2=complex(*v2)
    try:
        return math.degrees(cmath.polar(v2/v1)[1])
    except ZeroDivisionError:
        return 0

def try_to(f, args=None, kwargs=None, max_try='inf', exceptions=(KeyError,ValueError), raises=True):
    if max_try == 'inf':
        max_try = -1
    if args is None:
        args=[]
    if kwargs is None:
        kwargs ={}
    while True:
        try:
            return f(*args, **kwargs)
        except Exception as e:
            if max_try != 0 and any(map(lambda x: isinstance(e, x), exceptions)):
                max_try -= 1
                print(isinstance(e,exceptions[0]))
            else:
                if not raises:
                    print(e)
                raise

import glob

def load_images(folder,size=(48,48),mode='color'):
    files = glob.glob(os.path.join(folder,'*'))
    x_train=[]
    for file in files:
        image = imread(file,mode)
        image = Color.convert(image,'rgb')
        gray=cv2.resize(image,size)
        x_train.append(gray)
    x_train=np.array(x_train)
    x_train=x_train.reshape(len(x_train),size[0],size[1],1 if mode=='gray' else 3)    
    return x_train 




