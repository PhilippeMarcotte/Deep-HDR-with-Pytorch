from ModelUtilities import CropBoundariesMulti, CropBoundariesSingle, extract_patches_from_image, LDR_to_HDR, LDR_to_LDR
import Constants
import numpy as np
import numbers
from OpticalFlow import ComputeOpticalFlow

def ComputeTrainingExamples(imgs, expoTimes, label):

    imgs, label = PrepareInputFeatures(imgs, expoTimes, label)

    print(imgs.shape)
    imgs = CropBoundariesMulti(imgs, Constants.crop)
    print(label.shape)
    label = CropBoundariesSingle(label, Constants.crop)

    #Not sure if need to create empty arrays, we'll see
    augmentations = np.random.permutation(range(Constants.nbreTotalAugmentations))

    for i in range(0, Constants.nbreAugmentations - 1):
        augmentation = augmentations[i]
        #imgs, label = AugmentData(imgs, label, augmentation)

        #GetPatches
        # TODO : Decommenter quand extract_patches_from_image est fixed.
        #extract_patches_from_image(imgs, Constants.patchSize, Constants.batch_size)

        #Array with only 0 except for patch

    return  imgs, label

def PrepareInputFeatures(imgs, expoTimes, label):
    imgs = ComputeOpticalFlow(imgs, expoTimes)

    nanIndicesImg1 = np.where(imgs[0] == 0)
    imgs[0][nanIndicesImg1] = LDR_to_LDR(imgs[1][nanIndicesImg1], expoTimes[1], expoTimes[0])

    nanIndicesImg3 = np.where(label == 0)
    imgs[2][nanIndicesImg3] = LDR_to_LDR(imgs[1][nanIndicesImg3], expoTimes[1], expoTimes[2])

    darkIndices = np.where(imgs < 0.5)
    # TODO : Fixe ca!
    #badIndices = (darkIndices & nanIndicesImg3) | (not darkIndices & nanIndicesImg1)
    #abel[badIndices] = LDR_to_HDR(imgs[1][badIndices], expoTimes[1])

    #concatenate inputs
    hdrImgs = LDR_to_HDR(imgs, expoTimes, Constants.gamma)

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