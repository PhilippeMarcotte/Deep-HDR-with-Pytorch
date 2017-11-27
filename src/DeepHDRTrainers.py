import torch
import TrainingConstants
from abc import ABC, abstractmethod
import os
from datetime import datetime
import itertools
from DeepHDRModels import *
from ModelUtilities import l2_distance
from ModelUtilities import psnr
import shutil
import Constants

class DeepHDRTrainer(ABC):
    def __init__(self, model_name=None, checkpoint_name=None, checkpoints_folder = "./checkpoints/"):
        self.cnn = self.__build_model__()
        self.cuda_device_index = torch.cuda.device_count() - 1
        if torch.cuda.is_available():
            self.cnn = self.cnn.cuda(self.cuda_device_index)

        self.checkpoints_folder = os.path.join(checkpoints_folder, model_name, "")

        self.starting_iteration = 0
        self.optimizer = torch.optim.Adam(self.cnn.parameters(), TrainingConstants.learning_rate)

        self.best_psnr = 0

        os.makedirs(self.checkpoints_folder, exist_ok=True)
        
        if checkpoint_name:
            checkpoint_name = self.checkpoints_folder + checkpoint_name
            if os.path.isfile(checkpoint_name):
                print("loading checkpoint '{}'".format(checkpoint_name))
                checkpoint = torch.load(checkpoint_name)
                self.starting_iteration = checkpoint['epoch']
                self.best_psnr = checkpoint['best_psnr']
                self.cnn.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                    .format(checkpoint_name, checkpoint['epoch']))
            else:
                print("no checkpoint found at '{}'".format(checkpoint_name))
                print("starting with no checkpoint")

    def train(self):
        assert self.cnn

        scenes = DeepHDRScenes(root=os.path.join(Constants.training_data_root, Constants.training_directory))
        scene_loader = torch.utils.data.DataLoader(scenes, shuffle=True)      

        it = iter(scene_loader)
        for iteration in range(self.starting_iteration, TrainingConstants.num_iterations):
            try:
                scene_imgs, scene_labels = it.next()
            except StopIteration:
                print("resetting dataloader")
                scene_loader = torch.utils.data.DataLoader(scenes, shuffle=True)
                it = iter(scene_loader)
                scene_imgs, scene_labels = it.next()
                
            patches = DeepHDRPatches(scene_imgs.squeeze(), scene_labels.squeeze())
            patches_loader = torch.utils.data.DataLoader(patches, batch_size=20, shuffle=True, pin_memory=True)
            for j, (imgs, labels) in enumerate(patches_loader):
                patches = Variable(imgs)
                labels = Variable(labels)

                if torch.cuda.is_available():
                    patches = patches.cuda(self.cuda_device_index)
                    labels = labels.cuda(self.cuda_device_index)
                        
                self.optimizer.zero_grad()

                output = self.cnn(patches)

                loss = l2_distance(output, labels)

                loss.backward()

                self.optimizer.step()

            if iteration % TrainingConstants.validation_frequency == 0:
                is_best = self.validating()
                self.__make_checkpoint__(iteration, is_best)
    
    def validating(self):
        scenes = DeepHDRScenes(root=os.path.join(Constants.training_data_root, Constants.test_directory))

        scene_loader = torch.utils.data.DataLoader(scenes)
        sum_psnr = 0
        for i, (scene_imgs, scene_labels) in enumerate(scene_loader):            
            patches = DeepHDRPatches(scene_imgs.squeeze(), scene_labels.squeeze())
            patches_loader = torch.utils.data.DataLoader(patches, batch_size=20)
            scene_outputs = []
            scene_cropped_labels = []
            for j, (imgs, labels) in enumerate(patches_loader):
                patches = Variable(imgs)
                labels = Variable(labels)

                if torch.cuda.is_available():
                    patches = patches.cuda(self.cuda_device_index)
                    labels = labels.cuda(self.cuda_device_index)
                        
                self.optimizer.zero_grad()

                output = self.cnn(patches)
                scene_outputs.append(output.data)
                scene_cropped_labels.append(labels.data)

            outputs = torch.cat(scene_outputs, 0)
            labels = torch.cat(scene_cropped_labels, 0)
            sum_psnr += psnr(outputs, labels)

        average_psnr = sum_psnr / scene_loader.__len__()

        print("validation psnr : {}".format(average_psnr))

        if self.best_psnr < average_psnr:
            self.best_psnr = average_psnr
            return True

        return False
            
            
    @abstractmethod
    def __build_model__(self):
        pass

    def __make_checkpoint__(self, iteration, is_best, filename='checkpoint_{}.pth'):
        checkpoint_datetime = str(datetime.now())
        filename = filename.format(checkpoint_datetime)
        
        state = {
            'iteration': iteration,
            'state_dict': self.cnn.state_dict(),
            'best_psnr': self.best_psnr,
            'optimizer' : self.optimizer.state_dict(),
        }
        
        torch.save(state, self.checkpoints_folder + filename)
        if is_best:
            shutil.copyfile(self.checkpoints_folder + filename, self.checkpoints_folder + 'model_best.pth')

class DirectDeepHDRTrainer(DeepHDRTrainer):
    def __init__(self, checkpoint=None, checkpoints_folder = "./checkpoints/"):
        super(DirectDeepHDRTrainer, self).__init__("Direct", checkpoint, checkpoints_folder)
    
    def __build_model__(self):
        return DirectDeepHDR()

class WeDeepHDRTrainer(DeepHDRTrainer):
    def __init__(self, checkpoint=None, checkpoints_folder = "./checkpoints/"):
        super(WeDeepHDRTrainer, self).__init__("WE", checkpoint, checkpoints_folder)
    
    def __build_model__(self):
        return WeDeepHDR()

class WieDeepHDRTrainer(DeepHDRTrainer):
    def __init__(self, checkpoint=None, checkpoints_folder = "./checkpoints/"):
        super(WieDeepHDRTrainer, self).__init__("WIE", checkpoint, checkpoints_folder)
    
    def __build_model__(self):
        return WieDeepHDR()
        

trainer = DirectDeepHDRTrainer()

trainer.train()