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

# quiver(X[::step,::step], Y[::step,::step],vxs[::step,::step], vys[::step,::step],color='g', angles='xy', scale_units='xy', scale=0.25)



a = cv2.resize(imread(sys.argv[1]),(512,512))
b = cv2.resize(imread(sys.argv[2]),(512,512))
a = im2gray(a)
b = im2gray(b)


def transform_dtcwt(ref, src):
    step=4
    ref_t = transform2d.forward(ref, nlevels=4)
    src_t = transform2d.forward(src, nlevels=4)
    reg = registration.estimatereg(src_t, ref_t)
    vxs, vys = registration.velocityfield(reg, ref.shape[:2], method='nearest')
    mesh = np.sqrt(vxs*vxs + vys * vys)
    return mesh


c = transform_dtcwt(a, b)
c*= (255/c.max())
c=c.astype('int')

c[c < c.max() * .2] = 0
d = plt.imshow(c, cmap='hot')

plt.colorbar(d)
plt.figure()
k = np.dstack([a, b, c ])
plt.imshow(k.astype('int'))
plt.show()
