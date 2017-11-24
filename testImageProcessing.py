
import numpy as np
import cv2
import os, glob
from ImagePreprocessing import ReadImages, ApplyGammaCorrection, ReadExpoTimes, Demosaic, WriteImages

def listAllFiles(folderName, extension):
	os.chdir(folderName)
	return glob.glob('**/EXTRA/001/*'+extension)

#folderName = '/usagers/ulvil/Bureau/Automne2017/INF8702/INF8702-TP5/'
folderName = '/home/uvilleneuve/Documents/INF8702/Projet/INF8702-TP5/'

allFiles = listAllFiles(folderName, '.tif')
print(allFiles)

imgs = ReadImages(allFiles)
#cv2.imshow('image', imgs[2])
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#print(ReadImages(allFiles))

#print(ApplyGammaCorrection(imgs))
#imgsAfterGamma = ApplyGammaCorrection(imgs)

#print(ReadExpoTimes(listAllFiles(folderName, '.txt')[0]))
#imgsExpoTimes = ReadExpoTimes(listAllFiles(folderName, '.txt')[0])

################################## ONLY DEMOSAIC IF PGM and remove mosaicing from demosaicing
#Demosaic(imgs)

WriteImages(imgs, allFiles)

'''
listOfFiles = dir(sprintf('%s\\*.pgm', sceneFolder));
numImages = size(listOfFiles, 1);
inputLDRs = cell(1, numImages);

for i = 1 : numImages
    Path = sprintf('%s\\%s\\%s', inputSceneFolder, sceneName, listOfFiles(i).name);
    inputLDRs{i} = imread(Path);
'''