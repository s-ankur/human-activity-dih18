from __future__ import print_function, division

import cmath
import math


def dist(a, b=None):
    if b is None:
        b = [0, 0]
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def closest(points, source, key=lambda x: x):
    points.sort(key=lambda x: dist(key(x), source))
    return points.pop(0)


def angle(v1, v2=(1, 0)):
    v1 = complex(*v1)
    v2 = complex(*v2)
    try:
        return math.degrees(cmath.polar(v2 / v1)[1])
    except ZeroDivisionError:
        return 0


def try_to(f, args=None, kwargs=None, max_try=-1, exceptions=(KeyError, ValueError), silent=True):
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
    exceptions = tuple(exceptions)
    while max_try:
        try:
            return f(*args, **kwargs)
        except exceptions as e:
            if not silent:
                print(repr(e))
