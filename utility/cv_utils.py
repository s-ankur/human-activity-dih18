from __future__ import print_function, division
import cv2
import numpy as np
from contextlib import contextmanager
from logging import info, warning
try:
    from .utils import *
except:
    from utils import *
import numpy

class HardwareError(Exception):
    """
    Custom Exception raised when External Hardware fails
    """
    pass


@contextmanager
def window(*args,**kwargs):
    cv2.namedWindow( *args,**kwargs )
    yield
    destroy_window(*args,**kwargs)

def destroy_window(*args,**kwargs):
    cv2.destroyWindow(*args,**kwargs)
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)      
class Color:
    @staticmethod
    def convert( image ,color_to,color_from ='bgr',):
        colorspace_name=('COLOR_'+color_from+'2'+color_to).upper()
        try:
            colorspace=getattr(cv2,colorspace_name)
        except AttributeError:
            raise ValueError("No such colorspace: %s"%color_to)
        return cv2.cvtColor(image,colorspace)

    @staticmethod
    def from_crop(image,colorspace=None,color_name='color',):
        image, bbox = try_to(imcrop, args=[image, color_name])
        return Color(image,colorspace)

    def __init__(self, image,colorspace=None):      
        self.color = np.zeros((2, 3))
        self.colorspace = colorspace
        if self.colorspace:
            image = Color.convert(image, colorspace)
        for i in range(3):
            self.color[0, i] = image[:, :, i].max()
            self.color[1, i] = image[:, :, i].min()
        
    def threshold(self, image, err=np.array([35,5,5])):
        if self.colorspace:
            image = Color.convert(image, self.colorspace)
        image=cv2.inRange(image, self.color[1, :] - err, self.color[0, :] + err)
        return image
    
    def __repr__(self):
        if not self.colorspace:
            colorspace='rgb'
        else:
            colorspace = self.colorspace
        return ("Mode :  "+ colorspace +'\n'+ str(self.color))

class Video:
    bbox = None
    released=False
    
    def __init__(self, source,resolution =[1280,720]):
        try:
            info("Starting VideoCapture")
            self.input_video = cv2.VideoCapture(source)
            info("VideoCapture Started")
        except:
            raise HardwareError("Video Not Connected")

        if resolution is not None or isinstance(source,str):
            self.input_video.set(3,resolution[0])
            self.input_video.set(4,resolution[1])
        self.input_video.grab()

    def release(self):
        if not self.released:
            self.input_video.release()
            self.released=True
            info("Video Capture Released")

    def set_roi(self,bbox=None):
        info("Setting Region of interest")
        if bbox is None:
            _,image=self.input_video.read()
            image,bbox=imcrop(image)
        self.bbox=bbox
        info ("Region of interest set as",bbox )

    def read(self):
        ret, image = self.input_video.read()
        if not ret:
            raise ValueError("Videostream Not Working/ Existing")
        if self.bbox is not None:
            image,bbox=imcrop(image,bbox=self.bbox)
        return image

    def __iter__(self):
        try:
            while True:
                yield self.read()
        except ValueError:
            return 

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.release()

    def __del__(self):
        self.release()

def imshow(image,window_name='image',hold=False,):
    if not hold:
        cv2.namedWindow( window_name )
    # if image.shape[0]>700:
    #     warning("Image size too large, resizing to fit")
    #     image = cv2.resize(image, (0,0), fx=700/image.shape[0], fy=700/image.shape[0])  
    # if image.shape[1]>700:
    #     warning("Image size too large, resizing to fit")        
    #     image = cv2.resize(image, (0,0), fx=700/image.shape[1], fy=700/image.shape[1])     
    cv2.imshow(window_name,image)
    key = cv2.waitKey(int(hold)) & 0xFF 
    if not hold:
        destroy_window(window_name)
    return chr(key)

class _imtool:
    """
    MATLAB like imtool with very limited functionality
    Show color values and position at a given point in an image, interactively
    some problems when resizing
    """
    def __init__(self,image):
        self.image=image
        self.pos=(0,0)
        with window( 'imtool' ):
            cv2.setMouseCallback('imtool', self.on_click)    
            font=cv2.FONT_HERSHEY_SIMPLEX
            while True:
                image=np.zeros_like(self.image)
                x,y=self.pos
                cols=self.image[y,x]
                text="%d %d: "%(y,x)+str(cols)
                cv2.putText(image,text,self.pos,font,.5, 255)
                key = imshow(cv2.bitwise_xor(image,self.image),window_name='imtool',hold=True)
                if key == 'q':
                    break
                try:
                    cv2.getWindowProperty('imtool', 0)
                except cv2.error:
                    break   
        
    def on_click(self, event, x, y, flags, param):           
        self.pos=(x,y)

def imtool(image):
    _imtool(image)

class imcrop:
    """
    MATLAB-like imcrop utility
    Drag mouse over area to select
    Lift to complete selection
    Doubleclick or close window to finish choosing the crop
    Rightclick to retry
    
    Example:
        >>> cropped_img,bounding_box = imcrop(image)  # cropped_img is the cropped part of the image
    
        >>> crp_obj=imcrop(image,'img_name')          # asks for interactive crop and returns an imcrop object
        <imcrop object at ...>
        
        >>> crp_obj.bounding_box                    # the bounding_box of the crop
        [[12, 15] , [134 , 232]]
        
        >>> image,bbox=imcrop(image,bbox)               # without interactive cropping
        
    """
    modes = {
        'standby': 0,
        'cropping': 1,
        'crop_finished': 2,
        'done_exit': 3}

    def __init__(self, image, window_name='image',bbox=None,):
        self.window_name =  window_name
        self.image=image
        if  bbox is None:
            self.bounding_box = []
            self.mode = 0
            self.crop()
        else:
            self.bounding_box=bbox

    def crop(self):
        cv2.namedWindow( self.window_name);
        cv2.setMouseCallback(self.window_name, self.on_click)
        while True:
            img2 = self.image.copy()
            if self.mode > 0:
                cv2.rectangle(img2, self.bounding_box[0], self.current_pos, (0, 255, 0), 1)
            key = imshow(img2,window_name=self.window_name,hold=True)
            try:
                cv2.getWindowProperty(self.window_name, 0)
            except cv2.error:
                break
            if self.mode == 3:
                break
        destroy_window(self.window_name)
        if len(self.bounding_box) != 2 or self.bounding_box[0][0] == self.bounding_box[1][0] or self.bounding_box[0][1] == self.bounding_box[1][1]:
            raise ValueError("Insufficient Points selected")

    def __iter__(self):
        bbox=self.bounding_box
        if bbox[0][0] > bbox[1][0]:
            bbox[1][0], bbox[0][0] = bbox[0][0], bbox[1][0]
        if bbox[0][1] > bbox[1][1]:
            bbox[1][1], bbox[0][1] = bbox[0][1], bbox[1][1]
        yield self.image[int(bbox[0][1]):int(bbox[1][1]), int(bbox[0][0]):int(bbox[1][0])]
        yield bbox

    def on_click(self, event, x, y, flags, param):
        if self.mode == 0 and event == cv2.EVENT_LBUTTONDOWN:
            self.mode = 1
            self.current_pos = (x, y)
            self.bounding_box = [(x, y)]
        elif self.mode == 1 and event == cv2.EVENT_LBUTTONUP:
            self.mode = 2
            self.bounding_box.append((x, y))
            self.current_pos = (x, y)
        elif self.mode == 1 and event == cv2.EVENT_MOUSEMOVE:
            self.current_pos = (x, y)
        elif self.mode == 2 and event == cv2.EVENT_RBUTTONDOWN:
            self.mode = 0
        elif self.mode == 2 and event == cv2.EVENT_LBUTTONDBLCLK:
            self.mode = 3


_imread_modes={
    'color':cv2.IMREAD_COLOR,
    'gray':cv2.IMREAD_GRAYSCALE,
    'alpha':-1
    }

def imread(img_name,mode='color'):
    image=cv2.imread(img_name,_imread_modes[mode])
    if image is None:
        raise IOError(img_name)
    return image

imwrite=cv2.imwrite

_kernel_shapes={
    'rectangle':cv2.MORPH_RECT,
    'square':   cv2.MORPH_RECT,
    'circle':   cv2.MORPH_ELLIPSE,
    'ellipse':   cv2.MORPH_ELLIPSE,
    'cross':    cv2.MORPH_CROSS
    } 

def _kernel(kernel_name,size):
    return cv2.getStructuringElement(_kernel_shapes[kernel_name],size)

def imdilate(image,kernel='circle',size=5,iterations=1):
    return cv2.dilate(image.copy(),_kernel(kernel,(size,size)),iterations = iterations)

def imerode(image,kernel='circle',size=5,iterations=1):
    return cv2.erode(image.copy(),_kernel(kernel,(size,size)),iterations = iterations)

def imopen(image,kernel='circle',size=5):
    return cv2.morphologyEx(image.copy(), cv2.MORPH_OPEN, _kernel(kernel,(size,size)))

def imclose(image,kernel='circle',size=5):
    return cv2.morphologyEx(image.copy(), cv2.MORPH_CLOSE, _kernel(kernel,(size,size)))

def centroid(contour):
    m = cv2.moments(contour)
    cx = int(m["m10"] / m["m00"]) 
    cy = int(m["m01"] / m["m00"])
    return np.array((cy,cx))     

def polylines(image,points,closed=False,color=(0,255,0),show_points=True):
    image=image.copy()
    if show_points:
        for point in points:
            point=tuple(map(int,point))
            cv2.circle(image,point, 2, (0,0,255), -1)
    pts = np.array(points, np.int32); 
    pts = pts.reshape((-1,1,2));
    return cv2.polylines(image,[pts],closed,color)
    
def rectangle(image,corner1,corner2,color=(255,255,255),linewidth=-1):
    cv2.rectangle(image,corner1,corner2,color,linewidth)

def find_shapes(image,show=True):
    shapes=defaultdict(list)
    contours = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.1 * peri, True)
        if len(approx) == 3:
            shape_name = "triangle"
        elif len(approx) == 4:
            shape_name = "square"
        elif len(approx) == 5:
            shape_name = "pentagon"
        elif len(approx) == 6:
            shape_name = "hexagon"
        else:
            shape_name = "circle"
        shape=cv2.fillPoly(np.zeros_like(image), pts =[approx], color=(255,255,255))
        if show:
            imshow(shape,window_name=shape_name)
        shapes[shape_name].append(shape) 
    return shapes

def im2bw(image,otsu=True,threshold=127):   
    image=im2gray(image)
    (thresh, im_bw) = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY | (cv2.THRESH_OTSU * int(otsu)))
    return im_bw
    
def im2gray(image):
        return Color.convert(image,'gray')

def bbox2rect(bbox):
    (x,y,w,h)=bbox
    return x,y,x+w,y+h

def crop(image,bbox,extend=0):
    (x,y,w,h)=bbox
    return image[y-extend:y+h+extend,x-extend:x+w+extend]

def overlay(target,source):
    target[:,:,:] = cv2.resize(source,(target.shape[:-1][::-1]))

def blend_overlay(target,source):
    mask = im2bw(target, otsu=True, threshold=10)
    mask = (mask == 0)
    source=cv2.resize(source,target.shape[:2][::-1])
    target[mask] =  source[mask]

def blend_transparent(target,source):
    overlay=cv2.resize(source,target.shape[:2])
    source = overlay[:,:,:3] 
    overlay_mask = overlay[:,:,-1]
    background_mask = 255 - overlay_mask
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)
    face_part = (target * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (source * (1 / 255.0)) * (overlay_mask * (1 / 255.0))
    target[:,:,:] = np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))
    
def random_transform( image, rotation_range, zoom_range, shift_range, random_flip ):
    h,w = image.shape[0:2]
    rotation = numpy.random.uniform( -rotation_range, rotation_range )
    scale = numpy.random.uniform( 1 - zoom_range, 1 + zoom_range )
    tx = numpy.random.uniform( -shift_range, shift_range ) * w
    ty = numpy.random.uniform( -shift_range, shift_range ) * h
    mat = cv2.getRotationMatrix2D( (w//2,h//2), rotation, scale )
    mat[:,2] += (tx,ty)
    result = cv2.warpAffine( image, mat, (w,h), borderMode=cv2.BORDER_REPLICATE )
    if numpy.random.random() < random_flip:
        result = result[:,::-1]
    return result

def random_warp( image ):
    assert image.shape == (256,256,3)
    range_ = numpy.linspace( 128-80, 128+80, 5 )
    mapx = numpy.broadcast_to( range_, (5,5) )
    mapy = mapx.T
    mapx = mapx + numpy.random.normal( size=(5,5), scale=5 )
    mapy = mapy + numpy.random.normal( size=(5,5), scale=5 )
    interp_mapx = cv2.resize( mapx, (80,80) )[8:72,8:72].astype('float32')
    interp_mapy = cv2.resize( mapy, (80,80) )[8:72,8:72].astype('float32')
    warped_image = cv2.remap( image, interp_mapx, interp_mapy, cv2.INTER_LINEAR )
    src_points = numpy.stack( [ mapx.ravel(), mapy.ravel() ], axis=-1 )
    dst_points = numpy.mgrid[0:65:16,0:65:16].T.reshape(-1,2)
    mat = umeyama( src_points, dst_points, True )[0:2]
    target_image = cv2.warpAffine( image, mat, (64,64) )
    return warped_image, target_image

def umeyama( src, dst, estimate_scale ):
    num = src.shape[0]
    dim = src.shape[1]
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean
    A = np.dot(dst_demean.T, src_demean) / num
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1
    T = np.eye(dim + 1, dtype=np.double)
    U, S, V = np.linalg.svd(A)
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = np.dot(U, V)
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
            d[dim - 1] = s
    else:
        T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V.T))
    if estimate_scale:
        scale = 1.0 / src_demean.var(axis=0).sum() * np.dot(S, d)
    else:
        scale = 1.0
    T[:dim, dim] = dst_mean - scale * np.dot(T[:dim, :dim], src_mean.T)
    T[:dim, :dim] *= scale
    return T


import os

def load_images(folder,size=(48,48),mode='color'):
    files = os.listdir(folder)
    x_train=[]
    for file in files:
        image = imread(file,mode)
        image = Color.convert(image,'rgb')
        gray=cv2.resize(image,size)
        x_train.append(gray)
    x_train=np.array(x_train)
    x_train=x_train.reshape(len(x_train),size[0],size[1],1 if mode=='gray' else 3)    
    return x_train 


