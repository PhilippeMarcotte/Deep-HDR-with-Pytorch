import os,glob
from ImagePreprocessing import ReadExpoTimes, ReadTrainingData
from ComputeTrainingExamples import ComputeTrainingExamples

#TODO : Mettre dans un fichier Utilities
#Read training folder content (scenePaths et nombre de scenes)
def listAllFiles(folderName):
	#os.chdir(folderName)
	return glob.glob(folderName)

allSceneFolders = listAllFiles(Constants.folderName+'/*/')
print(allSceneFolders)
print(len(allSceneFolders))

for scene in allSceneFolders :
    #Read Expo times in scene
    expoTimes = ReadExpoTimes(scene+'exposure.txt')
    print(expoTimes)

    #Read Image in scene
    fileNames = listAllFiles(scene+'/*.tif')
    imgs, label = ReadTrainingData(fileNames)

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