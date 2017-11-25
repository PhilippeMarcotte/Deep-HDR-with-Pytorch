import os,glob
from ImagePreprocessing import ReadExpoTimes, ReadTrainingData
from ComputeTrainingExamples import ComputeTrainingExamples
import Constants
import numpy as np

#TODO : Mettre dans un fichier Utilities
#Read training folder content (scenePaths et nombre de scenes)
def listAllFiles(folderName):
	#os.chdir(folderName)
	return sorted(glob.glob(folderName))

allSceneFolders = np.array(listAllFiles(Constants.folderName+'/*/'))
print(allSceneFolders)
print(len(allSceneFolders))
print(allSceneFolders.shape)

for scene in allSceneFolders :
    #Read Expo times in scene
    expoTimes = ReadExpoTimes(scene+'exposure.txt')
    print(expoTimes)

    #Read Image in scene
    fileNames = listAllFiles(scene+'/*.tif')
    imgs, label = ReadTrainingData(fileNames)

    imgs = imgs/255
    # TODO : Remettre entre 0 et 255 avant d'ecrire la matrice

    #ComputeTraining examples in scene
    computed, computedLabel = ComputeTrainingExamples(imgs, expoTimes, label)

#Write in scene
# Faire plus tard car matiere a changements
# Utiliser nparray.toFile - Binary
#Output Format :
#--001--imgs
#      --label
#--002--imgs
#      --label
#--...--imgs
#      --label
#--074--imgs
#      --label