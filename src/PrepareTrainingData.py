import os,glob
from ImagePreprocessing import ReadExpoTimes, ReadTrainingData
from ComputeTrainingExamples import ComputeTrainingExamples
import Constants
import numpy as np
from joblib import Parallel, delayed
import multiprocessing

def listAllFolder(folderName):
	return next(os.walk(folderName))[1]

def listAllFiles(folderName, extension = ""):
	return glob.glob(os.path.join(folderName, "*" + extension))

scene_root = Constants.scenes_root
training_scene_directory = os.path.join(scene_root, Constants.training_directory, "")
test_scene_directory = os.path.join(scene_root, Constants.test_directory, "")    

training_data_root = Constants.training_data_root
train_set_training_data_directory = os.path.join(training_data_root, Constants.training_directory, "")
test_set_training_data_directory = os.path.join(training_data_root, Constants.test_directory, "")

def prepare_training_data(params):
    scene = params[0]
    is_training_set = params[1]    
    if is_training_set:
        training_data_directory = train_set_training_data_directory
        scene_directory = training_scene_directory
        scene_type = "training"
    else:
        training_data_directory = test_set_training_data_directory
        scene_directory = test_scene_directory
        scene_type = "test"        
    
    print("Process {} : processing {} scene {}".format(os.getpid(), scene_type, scene))

    os.makedirs(training_data_directory, exist_ok=True)     
    #Read Expo times in scene
    expoTimes = ReadExpoTimes(os.path.join(scene_directory, scene, 'exposure.txt'))
    print(expoTimes)

    #Read Image in scene
    fileNames = listAllFiles(os.path.join(scene_directory, scene), '.tif')
    imgs, label = ReadTrainingData(fileNames)

    #ComputeTraining examples in scene
    computed, computedLabel = ComputeTrainingExamples(imgs, expoTimes, label)

    training_data_scene_directory = os.path.join(training_data_directory, scene, "")

    os.makedirs(training_data_scene_directory, exist_ok=True)

    computed.astype('float32').tofile(os.path.join(training_data_scene_directory, "patches"))
    computedLabel.astype('float32').tofile(os.path.join(training_data_scene_directory, "label"))

    print("Process {} : {} scene {} DONE!".format(os.getpid(), scene_type, scene))

def distribute_training_data_preparation():
    training_scenes = listAllFolder(training_scene_directory)
    test_scenes = listAllFolder(test_scene_directory)
    training_parameters = np.ones(len(training_scenes))
    test_parameters = np.zeros(len(test_scenes))

    is_training_set_params = np.concatenate((training_parameters,test_parameters))

    scenes = np.concatenate((training_scenes,test_scenes))

    parameters = zip(scenes, is_training_set_params)

    #Parallel(n_jobs=-1)(delayed(prepare_training_data)(scene, is_training_set) for (scene, is_training_set) in parameters)

    pool = multiprocessing.pool.Pool(processes=2)

    results = pool.imap_unordered(prepare_training_data, parameters)

    for result in results:
        print(result)
    '''
    num_scenes_to_process = round(num_scenes/num_threads)

    threads = []
    for i in range(num_threads):
        if (i == num_threads - 1):
            num_scenes_to_process = num_scenes - scene
        thread = Thread(target=preprocess, args=(scene, num_scenes_to_process))
        thread.start()
        threads.append(thread)
        scene += num_scenes_to_process

    for thread in threads:
        thread.join()
    '''

if __name__ == "__main__":
    distribute_training_data_preparation()