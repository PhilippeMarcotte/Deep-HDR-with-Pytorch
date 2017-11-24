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
    def __init__(self, num_times):
        self.num_times = num_times

    def __call__(self, img):
        return np.rot90(img, self.num_times, (-2, -1))

class Fliplr(object):
    def __call__(self, img):
        return np.fliplr(img)

class Flipud(object):
    def __call__(self, img):
        return np.flipud(img)

class SwapColorChannels(object): 
    def __init__(self, color_channels_order):
        self.color_channels_order = color_channels_order

    def __call__(self, img):
        return img[self.color_channels_order, :, :]

class ImageAugmentation():
    def __init__(self):
        rotations = [Rotate90(i) for i in range(4)]

        flips = [Fliplr(), Flipud()]   

        color_channels_orders = set(permutations([0, 1, 2]))
        color_channel_swaps = [SwapColorChannels(color_channels_order) 
                                    for color_channels_order in color_channels_orders]

        self.transformations = []
        for i in range(len(color_channel_swaps)):
            for j in range(len(rotations)):
                for k in range(len(flips)):
                    self.transformations.append(Compose([
                                                    color_channel_swaps[i],
                                                    rotations[j],
                                                    flips[k]
                                                    ]))
    
    def augment(self, x, y):
        transformed_inputs = []
        transformed_labels = []
        for i in range(10):
            transformed_input = np.zeros(x.shape)
            for j in range(6):
                index = np.random.randint(0, len(self.transformations))
                transformed_input[j*3:(j+1)*3] = self.transformations[index](x[j*3:(j+1)*3])
               
            transformed_inputs.append(transformed_input)

            transformed_label = self.transformations[index](y)
            transformed_labels.append(transformed_label)

        return np.concatenate(transformed_inputs), np.concatenate(transformed_labels)
        