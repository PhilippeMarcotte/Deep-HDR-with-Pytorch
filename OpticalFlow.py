import numpy as np
import pyflow
from ModelUtilities import LDR_to_LDR

def ComputeOpticalFlow(imgs, expoTimes):
    warped = np.empty(imgs.shape)
    warped[1] = imgs[1]

    # Not sure if we should translate these lines
    #% local motion
    #v = ver;
    #havePar = any(strcmp('Parallel Computing Toolbox', {v.Name}));
    expoAdj = np.empty((2, 2, imgs[0].shape[0], imgs[0].shape[1], imgs[0].shape[2]))
    expoAdj[0] = AdjustExposure(imgs[0:2], expoTimes[0:2])
    expoAdj[1] = AdjustExposure(imgs[1:3], expoTimes[1:3])
    # TODO : S'assurer que cest la bonne manipulation
    expoAdj[1] = np.flip(expoAdj[1], 0)

    flow = np.empty(2)

    for i in range(0,1):
        flow[i] = ComputeCeLiu(expoAdj[i][1], expoAdj[i][1])

    #warped[0] = WarpUsingFlow(imgs[0], flow[0]);
    #warped[2] = WarpUsingFlow(imgs[2], flow[1]);

    return warped

def AdjustExposure(imgs, expoTimes):
    numImgs = imgs.shape[0]
    numExposures = expoTimes.shape

    assert(numImgs == numExposures, 'The number of input images is not equal to the number of exposures');

    adjusted = np.empty((numImgs, imgs.shape[1], imgs.shape[2], imgs.shape[3]))
    maxExpo = expoTimes.max()

    for imgInd in range(0, numImgs):
        adjusted[imgInd] = LDR_to_LDR(imgs[imgInd], expoTimes[imgInd], maxExpo)

    return adjusted

def ComputeCeLiu(target, source):
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    params = [alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations]

    print("Test")
    print(target)
    print("Test2")
    print(source)
    vx, vy, _ = pyflow.coarse2fine_flow(target, source, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations)
    print(vx)
    print(vy)

    #TODO : Should we do single(flow)?
    flow = np.concatenate((vx, vy), 2)
    return flow

def WarpUsingFlow(img1, img2):
    return

def isPixelNaN(img):
    # Numpy isNan : retourne vecteur de meme dim, avec 0 ou 1 si non-NaN ou si NaN
    #Donc utiliser ca au lieu de cette methode
    return