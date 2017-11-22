from skimage.util.shape import view_as_windows
#import torch
import numpy as np
    
def extract_patches_from_image(img, patch_size, stride):
    #TODO : Modifer pour que ca prennent un array d imgs instead, et retourne un array de batchs
    # TODO : After debug, si l'array a 4 dimension, exemple : 3*1500*100*3, il fait des windows de (patch_size*patch_size*patch_size*patch_size)
    # Donc il faut appeler view_as_windows, pour chaque image de l'array d'images et pour chaque couleurs.
    # Expect ouput to be huge since its gonna be 1480 * 980 pour chaque image, pour chaque couleur
    return view_as_windows(img, patch_size, stride)

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

def l2_distance(result, target):
        assert result.size() == target.size()
        return (target - result).pow(2).sum()

def tone_map(x):
    return torch.log(x.mul(mu).add(1)) / log(1 + mu)

def CropBoundariesMulti(imgs, cropSize):
    return imgs[:, cropSize : -cropSize + 1, cropSize : -cropSize + 1, :]

def CropBoundariesSingle(img, cropSize):
    return img[cropSize : -cropSize + 1, cropSize : -cropSize + 1, :]