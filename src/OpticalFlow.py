import numpy as np
import pyflow
from ModelUtilities import LDR_to_LDR
from scipy.ndimage.interpolation import map_coordinates
from scipy.interpolate import griddata
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.interpolate import interp2d
from scipy.interpolate import RectBivariateSpline
from joblib import Parallel, delayed
import multiprocessing
import scipy.io as io
from skimage.transform import warp

def ComputeOpticalFlow(imgs, expoTimes):
    warped = np.empty(imgs.shape)
    warped[1] = imgs[1]

    expoAdj = np.empty((2, 2, imgs[0].shape[0], imgs[0].shape[1], imgs[0].shape[2]))
    expoAdj[0] = AdjustExposure(imgs[0:2], expoTimes[0:2])
    expoAdj[1] = AdjustExposure(imgs[1:3], expoTimes[1:3])

    expoAdj[1] = np.flip(expoAdj[1], 0)
    
    flows = Parallel(n_jobs=-1)(delayed(ComputeCeLiu)(expoAdj[i][1], expoAdj[i][0]) for i in range(2))

    warped[0] = WarpUsingFlow(imgs[0], flows[0])
    warped[2] = WarpUsingFlow(imgs[2], flows[1])

    return warped

def AdjustExposure(imgs, expoTimes):
    numImgs = imgs.shape[0]
    numExposures = expoTimes.shape

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

    vx, vy, _ = pyflow.coarse2fine_flow(target, source, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations)

    flow = np.stack((vx, vy))
    flow = np.swapaxes(np.swapaxes(flow, 0, 2), 0, 1)
    return flow

def WarpUsingFlow(imgs, flows):
    hi = imgs.shape[0]
    wi = imgs.shape[1]
    c = imgs.shape[2]
    nbreImgs = 1

    hf = flows.shape[0]
    wf = flows.shape[1]

    hd = (hi - hf) / 2
    wd = (wi - wf) / 2

    warped = np.empty((hf, wf, c))

    X, Y = np.meshgrid(np.arange(0, wf), np.arange(0, hf))

    range_y = np.arange(wf)
    range_x = np.arange(hf)

    curX = X + flows[:, :, 0]
    curY = Y + flows[:, :, 1]

    curY_X = np.array([curY, curX])

    Y_X = np.array([Y,X])

    for i in range(0, c): 
        warped[:,:,i] = map_coordinates(imgs[:,:,i], curY_X, cval=np.nan)

    warped = np.clip(warped, 0, 1)
             
    return warped