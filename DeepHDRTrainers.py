import torch
import TrainingConstants
from abc import ABC, abstractmethod
import os
from datetime import datetime
import itertools
from DeepHDRModels import *
from ModelUtilities import l2_distance
from ModelUtilities import psnr

class DeepHDRTrainer(ABC):
    def __init__(self, checkpoint=None, checkpoints_folder = "./checkpoints/"):
        self.cnn = self.__build_model__()
        self.cuda_device_count = torch.cuda.device_count() - 1
        if torch.cuda.is_available():
            self.cnn = self.cnn.cuda(self.cuda_device_count)

        self.checkpoints_folder = os.path.join(checkpoints_folder, "training_started_{}".format(str(datetime.now())))

        if not os.path.exists(self.checkpoints_folder):
            os.makedirs(self.checkpoints_folder)

        self.starting_iteration = 0
        self.optimizer = torch.optim.Adam(self.cnn.parameters(), TrainingConstants.learning_rate)
        
        if checkpoint:
            if os.path.isfile(args.resume):
                print("loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                self.starting_iteration = checkpoint['iteration']
                best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
            else:
                print("no checkpoint found at '{}'".format(args.resume))
                print("starting normally")

    def train(self):
        assert self.cnn

        scenes = DeepHDRScenes(root="./Train/")
        scene_loader = torch.utils.data.DataLoader(scenes, shuffle=True)      

        it = iter(scene_loader)
        for iteration in range(self.starting_iteration, TrainingConstants.num_iterations):
            try:
                scene_imgs, scene_labels = it.next()
            except StopIteration:
                scene_loader = torch.utils.data.DataLoader(scenes, shuffle=True)
                it = iter(scene_loader)
                scene_imgs, scene_labels = it.next()
                
            patches = DeepHDRPatches(scene_imgs.squeeze(), scene_labels.squeeze())
            patches_loader = torch.utils.data.DataLoader(patches, batch_size=20, shuffle=True)
            for j, (imgs, labels) in enumerate(patches_loader):
                patches = Variable(imgs)
                labels = Variable(labels)

                if torch.cuda.is_available():
                    patches = patches.cuda(self.cuda_device_count)
                    labels = labels.cuda(self.cuda_device_count)
                        
                self.optimizer.zero_grad()

                output = self.cnn(patches)

                loss = l2_distance(output, labels)

                print(loss.data[0])

                loss.backward()

                self.optimizer.step()

            if iteration % TrainingConstants.validation_frequency == 0:
                is_best = self.validating()
                self.__make_checkpoint__(iteration, is_best)                
    
    def validating(self):
        scenes = DeepHDRScenes(root="./Validation/")

        scene_loader = torch.utils.data.DataLoader(scenes)
        sum_psnr = 0
        for i, (scene_imgs, scene_labels) in enumerate(scene_loader):            
            patches = DeepHDRPatches(scene_imgs.squeeze(), scene_labels.squeeze())
            patches_loader = torch.utils.data.DataLoader(patches, batch_size=20)
            for j, (imgs, labels) in enumerate(patches_loader):
                patches = Variable(imgs)
                labels = Variable(labels)

                if torch.cuda.is_available():
                    patches = patches.cuda(self.cuda_device_count)
                    labels = labels.cuda(self.cuda_device_count)
                        
                self.optimizer.zero_grad()

                output = self.cnn(patches)

                loss = l2_distance(output, labels)

                print(loss.data[0])

                sum_psnr += psnr(output, labels)
            
        average_psnr = sum_psnr / 45                

        return False
            
            
    @abstractmethod
    def __build_model__(self):
        pass

    def __make_checkpoint__(self, iteration, is_best, filename='checkpoint_{}.pth'):
        checkpoint_datetime = str(datetime.now())
        filename = filename.format(checkpoint_datetime)
        
        state = {
            'epoch': iteration,
            'state_dict': self.cnn.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : self.optimizer.state_dict(),
        }
        
        torch.save(state, self.checkpoints_folder + filename)
        if is_best:
            shutil.copyfile(self.checkpoints_folder + filename, self.checkpoints_folder + 'model_best.pth')

class DirectDeepHDRTrainer(DeepHDRTrainer):
    def __init__(self, checkpoint=None, checkpoints_folder = "./checkpoints/"):
        super(DirectDeepHDRTrainer, self).__init__(checkpoint, checkpoints_folder)
    
    def __build_model__(self):
        return DirectDeepHDR()

class WeDeepHDRTrainer(DeepHDRTrainer):
    def __init__(self, checkpoint=None, checkpoints_folder = "./checkpoints/"):
        super(WeDeepHDRTrainer, self).__init__(checkpoint, checkpoints_folder)
    
    def __build_model__(self):
        return WeDeepHDR()

class WieDeepHDRTrainer(DeepHDRTrainer):
    def __init__(self, checkpoint=None, checkpoints_folder = "./checkpoints/"):
        super(WieDeepHDRTrainer, self).__init__(checkpoint, checkpoints_folder)
    
    def __build_model__(self):
        return WieDeepHDR()
        

trainer = DirectDeepHDRTrainer()

trainer.train()