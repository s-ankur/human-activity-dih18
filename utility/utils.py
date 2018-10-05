from __future__ import print_function, division
import requests
import cmath
import math

from keras.utils import generic_utils


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


def download_file(url,file):
    response = requests.get(url, stream=True)
    total_length = int(response.headers.get('content-length'))
    progressbar = generic_utils.Progbar(total_length)
    with open(file, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            progressbar.add(len(chunk))
            f.write(chunk)
        progressbar.update(len(chunk))
