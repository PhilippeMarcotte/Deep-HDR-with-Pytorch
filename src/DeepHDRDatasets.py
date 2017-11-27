from torch.utils.data.dataset import Dataset
import numpy as np
import os
import torch
import math
import torchvision.transforms as transforms
import ModelsConstants

class DeepHDRScenes(Dataset):
    def __init__(self, root):
        self.root = os.path.join(root, '')

        self.scenes = next(os.walk(self.root))[1]

        self.num_patches = 10*49*74
        
    def __getitem__(self, index):
        scene = math.floor(index/self.num_patches)
        scene = self.scenes[scene]

        scene_labels = np.fromfile(os.path.join(self.root, scene, "label"), dtype='uint8')
        scene_labels = np.reshape(scene_labels, (40, 40, 3, -1))
        scene_labels = np.rollaxis(np.rollaxis(scene_labels, 3), 3, 1)
        scene_labels = np.squeeze(scene_labels)        
        
        scene_imgs = np.fromfile(os.path.join(self.root, scene, "imgs"), dtype='uint8')
        scene_imgs = np.reshape(scene_imgs, (40, 40, 3, -1))
        scene_imgs = np.rollaxis(np.rollaxis(scene_imgs, 3), 3, 1)
        scene_imgs = np.squeeze(scene_imgs)

        return (scene_imgs, scene_labels)

    def __len__(self):
        return len(self.scenes)

class DeepHDRPatches(Dataset):
    def __init__(self, scene_imgs, scene_labels):
        self.scene_imgs = scene_imgs
        self.scene_labels = scene_labels
        self.label_transforms = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.CenterCrop(ModelsConstants.cnn_ouput_size),
                        transforms.ToTensor(),
                        transforms.Lambda(lambda crop: crop.renorm(1, 0, 255))
                        ])
    
    def __getitem__(self, index):
        imgs = self.scene_imgs[index].float().renorm(1, 0, 255)

        label = self.scene_labels[index]
        label = self.label_transforms(label)
        return (imgs, label)

    def __len__(self):
        return self.scene_imgs.shape[0]