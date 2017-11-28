from skimage.util.shape import view_as_windows
import torch
import numpy as np
import Constants
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

def weighted_average(weights, imgs, num_channels):
    assert weights.size() == imgs.size()
    assert weights.size()[1] % num_channels == 0
    
    mat_size = weights.size()
    num_imgs = mat_size[1] / num_channels

    average_denominator = torch.zeros(mat_size[0], 3, mat_size[2], mat_size[3])
    average_numerator = torch.zeros(mat_size[0], 3, mat_size[2], mat_size[3])
    for index in range(num_imgs):
        weight_mat = weights[:, index * num_channels:(index + 1) * num_channels]
        img = imgs[:, index * num_channels:(index + 1) * num_channels]

        average_denominator += weight_mat

        average_numerator += weight_mat.matmul(img)
    
    return average_numerator / average_denominator

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
    return torch.log(x.mul(ModelsConstants.mu).add(1)) / log(1 + ModelsConstants.mu)

def psnr(x, target):
    sqrdErr = torch.mean((x - target) ** 2)
    return 10 * log(1/sqrdErr)
    
# Je pense que ces fonctions crop pas la bonne taille. Je crois qu<il faut enlever le -1
# A voir
def CropBoundaries(imgs, cropSize):
    return imgs[cropSize : -cropSize, cropSize : -cropSize, :]

def crop_center(img,cropx,cropy):
    y,x = img.size()
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]
