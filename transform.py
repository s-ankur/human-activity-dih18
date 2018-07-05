import dtcwt
import dtcwt.registration as registration
import numpy as np
from utility.cv_utils import *

import matplotlib.pyplot as plt
import sys

transform2d = dtcwt.Transform2d()


# warped_src = registration.warp(src, reg, method='bilinear')
# vxs, vys = registration.velocityfield(reg, ref.shape[:2], method='bilinear')
# vxs = vxs*ref.shape[1]
# vys = vys*ref.shape[0]
# figure()
# X, Y = np.meshgrid(np.arange(ref.shape[1]), np.arange(ref.shape[0]))
# imshow(ref, cmap=cm.gray, clim=(0,1))
# step = 8

# quiver(X[::step,::step], Y[::step,::step],vxs[::step,:
# :step], vys[::step,::step],color='g', angles='xy', scale_units='xy', scale=0.25)


def transform_dtcwt(ref, src):
    ref_t = transform2d.forward(ref, nlevels=4)
    src_t = transform2d.forward(src, nlevels=4)
    reg = registration.estimatereg(src_t, ref_t)
    vxs, vys = registration.velocityfield(reg, ref.shape[:2], method='nearest')
    mesh = np.sqrt(vxs * vxs + vys * vys)
    return mesh


def dtcwt3d_layer(input_shape):
    f = lambda X: transform2d.forward(X, 2).lowpass
    if len(input_shape) == 3:
        f = lambda X: np.array([X[:, :, i] for i in range(input_shape[-1])])
        f = lambda X: f(X).reshape((1, input_shape[0] // 2, input_shape[1] // 2, input_shape[2]))
    else:
        f = lambda X: f(X).reshape((1, input_shape[0] // 2, input_shape[1] // 2))
    return Lambda(f)
