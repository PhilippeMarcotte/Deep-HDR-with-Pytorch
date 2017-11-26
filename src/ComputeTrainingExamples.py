from ModelUtilities import CropBoundaries, extract_patches_from_image, LDR_to_HDR, LDR_to_LDR
import Constants
import numpy as np
import numbers
from OpticalFlow import ComputeOpticalFlow
from DataAugmentation import ImageAugmentation
from ImagePreprocessing import select_subset
from ImagePreprocessing import get_num_patches

def ComputeTrainingExamples(imgs, expoTimes, label):

    imgs, label = PrepareInputFeatures(imgs, expoTimes, label)

    imgs = CropBoundaries(imgs, Constants.crop)
    label = CropBoundaries(label, Constants.crop)

    imgAugmentation = ImageAugmentation()

    num_patches = get_num_patches(imgs.shape[0], imgs.shape[1], Constants.patchSize, Constants.stride)

    imgs_patches = np.zeros((Constants.patchSize, Constants.patchSize, imgs.shape[-1], Constants.num_augmentations * num_patches), dtype='float32')
    labels_patches = np.zeros((Constants.patchSize, Constants.patchSize, label.shape[-1], Constants.num_augmentations * num_patches), dtype='float32')
    for i in range(Constants.num_augmentations):
        transformed_imgs, transformed_labels = imgAugmentation.augment(imgs, label)

        cur_imgs_patches = extract_patches_from_image(transformed_imgs, Constants.patchSize, Constants.stride)
        imgs_patches[:, :, :, i * num_patches:(i + 1) * num_patches] = cur_imgs_patches.astype('float32')

        cur_label_patches = extract_patches_from_image(transformed_labels, Constants.patchSize, Constants.stride)
        labels_patches[:, :, :, i * num_patches:(i + 1) * num_patches] = cur_label_patches.astype('float32')

    indexes = select_subset(imgs_patches[:, :, 3:6])

    imgs_patches = imgs_patches[:, :, :, indexes]

    labels_patches = labels_patches[:, :, :, indexes]

    return  imgs_patches, labels_patches

def PrepareInputFeatures(imgs, expoTimes, label):
    imgs = ComputeOpticalFlow(imgs, expoTimes)

    nanIndicesImg1 = np.isnan(imgs[0])
    imgs[0][nanIndicesImg1] = LDR_to_LDR(imgs[1][nanIndicesImg1], expoTimes[1], expoTimes[0])

    nanIndicesImg3 = np.isnan(imgs[2])
    imgs[2][nanIndicesImg3] = LDR_to_LDR(imgs[1][nanIndicesImg3], expoTimes[1], expoTimes[2])

    darkIndices = imgs[1] < 0.5
    dark_and_nan3 = np.logical_and(darkIndices, nanIndicesImg3)
    not_dark_and_nan1 = np.logical_and(np.logical_not(darkIndices), nanIndicesImg1)
    badIndices = np.logical_or(dark_and_nan3, not_dark_and_nan1)
    label[badIndices] = LDR_to_HDR(imgs[1][badIndices], expoTimes[1], Constants.gamma)

    #concatenate inputs
    hdrImgs = LDR_to_HDR(imgs, expoTimes, Constants.gamma)

    imgs = np.concatenate((imgs[0], imgs[1], imgs[2]), 2)
    hdrImgs = np.concatenate((hdrImgs[0], hdrImgs[1], hdrImgs[2]), 2)

    imgs = np.concatenate((imgs, hdrImgs), 2)
    return imgs, label

def GetNumPatches(width, height):
    return

#Do with numpy instead?
def GetPatches(imgs, patchSize, stride):
    return

# Necessaire?
def SelectSubset(imgs):
    return