import os
from ImagePreprocessing import ReadExpoTimes, ReadTrainingData
from ComputeTrainingExamples import ComputeTrainingExamples
import Constants
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from contextlib import closing
import h5py
from ModelUtilities import list_all_files_sorted
from ModelUtilities import list_all_folders
from tqdm import tqdm

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

    os.makedirs(training_data_directory, exist_ok=True)     
    #Read Expo times in scene
    expoTimes = ReadExpoTimes(os.path.join(scene_directory, scene, 'exposure.txt'))

    #Read Image in scene
    fileNames = list_all_files_sorted(os.path.join(scene_directory, scene), '.tif')
    imgs, label = ReadTrainingData(fileNames)

    #ComputeTraining examples in scene
    computed, computedLabel = ComputeTrainingExamples(imgs, expoTimes, label, is_training_set)

    hf = h5py.File(os.path.join(training_data_directory, scene+".data"), 'w')

    computed = np.rollaxis(np.rollaxis(computed, 3), 3, 1)
    computedLabel = np.rollaxis(np.rollaxis(computedLabel, 3), 3, 1)
    print("WRITING")
    hf.create_dataset("inputs", data=computed)
    hf.create_dataset("labels", data=computedLabel)

    hf.close()

def distribute_training_data_preparation():
    training_scenes = list_all_folders(training_scene_directory)
    already_processed_training_scenes = list_all_files_sorted(train_set_training_data_directory)
    already_processed_training_scenes = [os.path.basename(data).split('.')[0] for data in already_processed_training_scenes]
    training_scenes = set(training_scenes) - set(already_processed_training_scenes)

    test_scenes = list_all_folders(test_scene_directory)
    already_processed_test_scenes = list_all_files_sorted(test_set_training_data_directory)
    already_processed_test_scenes = [os.path.basename(data).split('.')[0] for data in already_processed_test_scenes]
    test_scenes = set(test_scenes) - set(already_processed_test_scenes)
    
    training_parameters = np.ones(len(training_scenes))
    test_parameters = np.zeros(len(test_scenes))

    if len(training_scenes) > 0 and len(test_scenes) > 0:
        scenes = np.concatenate((training_scenes,test_scenes))
    elif len(training_scenes) > 0:
        scenes = training_scenes
    else:
        scenes = test_scenes        

    is_training_set_params = np.concatenate((training_parameters,test_parameters))

    parameters = zip(scenes, is_training_set_params)

    #Parallel(n_jobs=-1)(delayed(prepare_training_data)(scene, is_training_set) for (scene, is_training_set) in parameters)
    with closing(multiprocessing.pool.Pool(processes=1, maxtasksperchild=1)) as pool:
        with tqdm(total=len(scenes)) as pbar:
            for i, _ in tqdm(enumerate(pool.imap_unordered(prepare_training_data, parameters))):
                pbar.update()

if __name__ == "__main__":
    distribute_training_data_preparation()
