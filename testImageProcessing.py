
import numpy as np
#import cv2
import os, glob

def listAllFiles(folderName, extension):
	os.chdir(folderName)
	return glob.glob('**/*'+extension)

folderName = '/usagers/ulvil/Bureau/Automne2017/INF8702/INF8702-TP5/'

allFiles = listAllFiles(folderName, '.tif')
print(allFiles)

#img = cv2.imread(fileName,0)
#cv2.showImg('image', img)


'''
listOfFiles = dir(sprintf('%s\\*.pgm', sceneFolder));
numImages = size(listOfFiles, 1);
inputLDRs = cell(1, numImages);

for i = 1 : numImages
    Path = sprintf('%s\\%s\\%s', inputSceneFolder, sceneName, listOfFiles(i).name);
    inputLDRs{i} = imread(Path);
'''