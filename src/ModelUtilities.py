from skimage.util.shape import view_as_windows
import torch
import numpy as np
import Constants
from math import log10
from math import log
import os, glob

def list_all_folders(folderName):
	return next(os.walk(folderName))[1]

def list_all_files_sorted(folderName, extension = ""):
	return sorted(glob.glob(os.path.join(folderName, "*" + extension)))

def extract_patches_from_image(img, patch_size, stride):
    [height, width, depth] = img.shape
    num_patches = (np.floor((width - patch_size) / stride) + 1) * (np.floor((height - patch_size) / stride) + 1)
    patches = np.zeros((patch_size, patch_size, depth, num_patches.astype('int')))

    count = 0
    for x in range(0, width - patch_size + 1, stride):
        for y in range(0, height - patch_size + 1, stride):            
            patches[:, :, :, count] = img[y:y + patch_size, x:x + patch_size, :]
            count += 1
    return patches

def weighted_average(weights, imgs, num_channels=3):
    assert weights.size()[1] % num_channels == 0

    w1 = weights[:, 0:3]
    w2 = weights[:, 3:6]
    w3 = weights[:, 6:9]

    im1 = imgs[:, 0:3]
    im2 = imgs[:, 3:6]
    im3 = imgs[:, 6:9]

    weights_sum = (w1 + w2 + w3).add(float(np.finfo(np.float32).eps))
    
    output = (im1 * w1 + im2 * w2 + im3 * w3) / weights_sum

    return output

def LDR_to_HDR(imgs, expo, gamma):
    return (imgs ** gamma) / expo

def LDR_to_LDR(img, expo, expo2):
    Radiance = LDR_to_HDR(img, expo, Constants.gamma)
    return HDR_to_LDR(Radiance, expo2)

def HDR_to_LDR(img, expo):
    img = img.astype('float32')
    img *= expo
    img = np.clip(img, 0, 1)
    img = img ** (1/Constants.gamma)
    return img

def l2_distance(result, target):
    assert result.size() == target.size()
    return (target - result).pow(2).sum()

def range_compressor(x):
    return (torch.log(x.mul(Constants.mu).add(1))) / log(1 + Constants.mu)

def psnr(x, target):
    sqrdErr = torch.mean((x - target) ** 2)
    return 10 * log10(1/sqrdErr)

def crop_boundaries(imgs, cropSize):
    return imgs[cropSize : -cropSize, cropSize : -cropSize, :]

def crop_center(img,crop):
    y = img.size()[-2]
    x = img.size()[-1]
    return img[:, :, crop:y - crop, crop:x - crop]