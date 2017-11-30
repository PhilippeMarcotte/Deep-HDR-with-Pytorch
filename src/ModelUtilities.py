from skimage.util.shape import view_as_windows
import torch
import numpy as np
import Constants
import ModelsConstants
from math import log10
from math import log
import os, glob

def list_all_folders(folderName):
	return next(os.walk(folderName))[1]

def list_all_files_sorted(folderName, extension = ""):
	return sorted(glob.glob(os.path.join(folderName, "*" + extension)))

def extract_patches_from_image(img, patch_size, stride):
    '''
    patches = view_as_windows(img, (patch_size, patch_size, img.shape[-1]), stride)
    patches = patches.reshape((-1, patch_size, patch_size, img.shape[-1]))
    patches = np.rollaxis(patches, axis=0, start=4)
    '''
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

    imgs = crop_center(imgs, ModelsConstants.cnn_ouput_size)

    im1 = imgs[:, 0:3]
    im2 = imgs[:, 3:6]
    im3 = imgs[:, 6:9]

    weights_sum = (w1 + w2 + w3).add(float(np.finfo(np.float32).eps))
    
    output = (im1 * w1 + im2 * w2 + im3 * w3) / weights_sum

    return output

def LDR_to_HDR(imgs, expo, gamma):
    return (imgs ** gamma) / expo

def LDR_to_LDR(img, expo, expo2):
    # TODO :
    Radiance = LDR_to_HDR(img, expo, Constants.gamma)
    return HDR_to_LDR(Radiance, expo2)

def HDR_to_LDR(img, expo):
    #TODO : Not sure if this line is needed
    #img = img.astype('float32')
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
    
# Je pense que ces fonctions crop pas la bonne taille. Je crois qu<il faut enlever le -1
# A voir
def CropBoundaries(imgs, cropSize):
    return imgs[cropSize : -cropSize, cropSize : -cropSize, :]

def crop_center(img,crop):
    y = img.size()[-2]
    x = img.size()[-1]
    startx = x // 2 - (crop // 2)
    starty = y // 2 - (crop // 2)
    return img[:, :, starty:starty + crop, startx:startx + crop]

if __name__ == "__main__":
    mat1 = torch.ones((1,3,40,40))
    mat2 = 2*torch.ones((1,3,40,40))
    mat3 = 3*torch.ones((1,3,40,40))
    weights = torch.ones((1,9,28,28))
    mat = torch.cat((mat1,mat2,mat3), 1)
    weighted_average(weights, mat)