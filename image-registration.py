#!/usr/bin/env python
"""
An example of image registration via the DTCWT.

This script demonstrates some methods for image registration using the DTCWT.

"""

from __future__ import division, print_function
from utility.cv_utils import *

import itertools
import logging
import os

from matplotlib.pyplot import *
import numpy as np

import dtcwt
from dtcwt.numpy import Transform2d
import dtcwt.sampling
from dtcwt.registration import *
logging.basicConfig(level=logging.INFO)


def register_frames(f1,f2):
    # Load test image

    # Take the DTCWT of both frames.
    logging.info('Taking DTCWT')
    nlevels = 6
    trans = Transform2d()
    t1 = trans.forward(f1, nlevels=nlevels)
    t2 = trans.forward(f2, nlevels=nlevels)

    # Solve for transform
    logging.info('Finding flow')
    avecs = estimatereg(t1, t2)

    logging.info('Computing warped image')
    warped_f1 = warp(f1, avecs, method='bilinear')

    logging.info('Computing velocity field')
    step = 16
    X, Y = np.meshgrid(np.arange(f1.shape[1]), np.arange(f1.shape[0]))
    vxs, vys = velocityfield(avecs, f1.shape, method='nearest')

    vxs -= np.median(vxs.flat)
    vys -= np.median(vys.flat)

    figure(figsize=(16,9))

    subplot(221)
    imshow(np.dstack((f1, f2, np.zeros_like(f1))))
    title('Overlaid frames')

    subplot(222)
    imshow(np.dstack((warped_f1, f2, np.zeros_like(f2))))
    title('Frame 1 warped to Frame 2 (image domain)')

    subplot(223)
    sc = 2
    imshow(np.dstack((f1, f2, np.zeros_like(f2))))
    quiver(X[::step,::step], Y[::step,::step],
            -sc*vxs[::step,::step]*f1.shape[1], -sc*vys[::step,::step]*f1.shape[0],
            color='b', angles='xy', scale_units='xy', scale=1)
    title('Computed velocity field (median subtracted), x{0}'.format(sc))

    subplot(224)
    imshow(np.sqrt(vxs*vxs + vys*vys), interpolation='none', cmap=cm.hot)
    colorbar()
    title('Magnitude of computed velocity (median subtracted)')
    show()

    # savefig(os.path.splitext(os.path.basename(filename))[0] + '-registration.png')

a = imread(sys.argv[1])
b = imread(sys.argv[2])
a = cv2.resize(im2gray(a),(1024,1024))
b = cv2.resize(im2gray(b),(1024,1024))

register_frames(a,b)
