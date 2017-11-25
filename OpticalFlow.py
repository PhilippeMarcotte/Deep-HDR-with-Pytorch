import numpy as np
import pyflow
from ModelUtilities import LDR_to_LDR
from scipy.interpolate import griddata

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

    flow = np.empty((2, 1000, 1500, 2))

    for i in range(0,1):
        flow[i] = ComputeCeLiu(expoAdj[i][1], expoAdj[i][0])

    warped[0] = WarpUsingFlow(imgs[0], flow[0])
    warped[2] = WarpUsingFlow(imgs[2], flow[1])

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

    #print("Test")
    #print(target)
    #print("Test2")
    #print(source)
    vx, vy, _ = pyflow.coarse2fine_flow(target, source, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations)
    #vx = 0
    #vy = 0
    print('vx :')
    print(vx.shape)
    print(vx)
    print('vy')
    print(vy.shape)
    print(vy)

    #TODO : Should we do single(flow)?
    #flow = np.concatenate((vx, vy), 2)
    flow = np.array([vx,vy])
    flow = np.swapaxes(np.swapaxes(flow, 0, 2), 0, 1)
    return flow

def WarpUsingFlow(imgs, flows):

    # TODO: Dont think we need this param
    #if (~exist('needBase', 'var') || isempty(needBase))
    #    needBase = true;
    #end

    # We decided we didnt need this
    #flows = gather(flows);
    print(imgs.shape)

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
    Xi, Yi = np.meshgrid(np.arange(wd, wf+wd), np.arange(hd, hf+hd))
    print("X Y")
    print(X)
    print(Y)
    print("Xi Yi")
    print(Xi)
    print(Yi)

    for i in range(0, c-1):
        '''
        if (needBase)
            curX = X + flows[:, :, 0]
            curY = Y + flows[:, :, 1]
        else
        '''
        curX = flows[:, :, 0]
        curY = flows[:, :, 1]
        #end
        
        #warped[:, :, i] = scipy.(Xi, Yi, imgs[:, :, i], curX, curY, 'cubic', nan)
        print("Xi, Yi, imgs, imgs[:,:,i], curX, curY shapes")
        print(Xi.shape)
        print(Yi.shape)
        Zi = (Xi, Yi)
        p = np.broadcast_arrays(*Zi)
        n = len(p)
        for j in range(1, n):
            if p[j].shape != p[0].shape:
                raise ValueError("coordinate arrays do not have the same shape")
        Zi = np.empty(p[0].shape + (len(Zi),), dtype=float)
        for j, item in enumerate(p):
            Zi[...,j] = item
        print(Zi.shape)
        print(Zi.shape[0])
        #print((Xi, Yi).shape)
        print(imgs[:,:,i].shape)
        print(imgs[:,:,i].shape[0])
        print(curX.shape)
        print(curY.shape)
        warped[:,:,i] = griddata((Xi, Yi), imgs[:,:,i], (curX, curY), method='cubic')

    warped = np.clip(warped, 0, 1);

    #%warped(isnan(warped)) = 0;
             
    return warped

def isPixelNaN(img):
    # Numpy isNan : retourne vecteur de meme dim, avec 0 ou 1 si non-NaN ou si NaN
    #Donc utiliser ca au lieu de cette methode
    return