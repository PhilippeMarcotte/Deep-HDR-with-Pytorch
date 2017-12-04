from torch.utils.data.dataset import Dataset
import numpy as np
import os
import torch
import math
import torchvision.transforms as transforms
import ModelsConstants
from ModelUtilities import list_all_files_sorted
import h5py
from ModelUtilities import range_compressor
from ModelUtilities import crop_center

class ScenesDeepHDR(Dataset):
    def __init__(self, root):
        self.root = os.path.join(root, '')

        scenes = list_all_files_sorted(self.root)
        self.hdf5_scenes = [h5py.File(scene, mode='r') for scene in scenes]

        self.label_transforms = transforms.Compose([
                        transforms.Lambda(lambda tensor: crop_center(tensor, ModelsConstants.cnn_ouput_size)),
                        transforms.Lambda(lambda crops: range_compressor(crops))])
        
    def __getitem__(self, index):
        scene = self.hdf5_scenes[index]

        scene_labels = torch.from_numpy(np.array(scene.get("labels")))
        scene_labels = self.label_transforms(scene_labels)
        
        scene_imgs = torch.from_numpy(np.array(scene.get("inputs")))

        return (scene_imgs, scene_labels, index)

    def __len__(self):
        return len(self.hdf5_scenes)

    def close(self):
        for scene in self.hdf5_scenes:
            scene.close()

class PatchesDeepHDR(Dataset):
    def __init__(self, scene_imgs, scene_labels):
        self.scene_imgs = scene_imgs
        self.scene_labels = scene_labels
    
    def __getitem__(self, index):
        imgs = self.scene_imgs[index]

        label = self.scene_labels[index]
        return (imgs, label)

    def __len__(self):
        return self.scene_imgs.size()[0]

class RefinerScenesDeepHDR(Dataset):
    def __init__(self, root):
        self.root = os.path.join(root, '')
        
        scenes = list_all_files_sorted(self.root)
        self.hdf5_scenes = [h5py.File(scene, mode='r') for scene in scenes]

        self.label_transforms = transforms.Compose([
                        transforms.Lambda(lambda tensor: crop_center(tensor, ModelsConstants.cnn_ouput_size))])
        
    def __getitem__(self, index):
        scene = self.hdf5_scenes[index]

        scene_imgs = torch.from_numpy(np.array(scene.get("inputs")))

        scene_labels = scene_imgs.clone()
        scene_labels = self.label_transforms(scene_labels)

        return (scene_imgs, scene_labels[:, 0:9], index)

    def __len__(self):
        return len(self.hdf5_scenes)

    def close(self):
        for scene in self.hdf5_scenes:
            scene.close()