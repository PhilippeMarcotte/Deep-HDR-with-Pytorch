from PIL import Image, ImageOps, ImageEnhance
import numpy as np
from itertools import permutations

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

class Rotate90(object):
    def __init__(self, num_times, ndim):
        self.num_times = num_times
        self.ndim = ndim

    def __call__(self, img):
        return np.rot90(img, self.num_times, (-(self.ndim - 1), -(self.ndim -2)))

class Fliplr(object):
    def __call__(self, img):
        return np.fliplr(img)

class Flipud(object):
    def __call__(self, img):
        return np.flipud(img)

class SwapColorChannels(object):
    def __init__(self, axes_order):
        self.axes_order = axes_order

    def __call__(self, img):
        return np.transpose(img, self.axes_order)

class ImageAugmentation():
    def __init__(self, axes_order):
        self.transforms = []
        