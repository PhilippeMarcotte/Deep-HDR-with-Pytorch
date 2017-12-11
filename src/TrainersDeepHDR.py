import torch
import TrainingConstants
from abc import ABC, abstractmethod
import os
from datetime import datetime
import itertools
from ModelsDeepHDR import *
from ModelUtilities import l2_distance
from ModelUtilities import psnr
import shutil
import Constants
from tqdm import tqdm
from contextlib import closing
import argparse
import glob
from ImagePreprocessing import ReadExpoTimes
from ModelUtilities import list_all_folders

class TrainerDeepHDR(ABC):
    def __init__(self, model_name=None, checkpoint=None, checkpoints_folder = "./checkpoints/"):
        self.cnn = self.__build_model__()     

        self.checkpoints_folder = os.path.join(checkpoints_folder, model_name, "")

        self.starting_iteration = 0
        self.optimizer = torch.optim.Adam(self.cnn.parameters(), TrainingConstants.learning_rate)

        self.best_psnr = 0

        os.makedirs(self.checkpoints_folder, exist_ok=True)
        self.psnr_track_file = os.path.join(self.checkpoints_folder, "psnrs.txt")        
        
        from_checkpoint = self.__load_checkpoint__(checkpoint)
        
        if torch.cuda.is_available():
            #if torch.cuda.device_count() > 1:
            #    self.cnn = torch.nn.DataParallel(self.cnn)
            self.cnn = self.cnn.cuda()
        
        if not from_checkpoint:
            with open(self.psnr_track_file, "w+") as f:
                f.write("")

    def __loss__(self, output, labels):
        return l2_distance(output, labels)

    def __train__(self, imgs, labels, **kwargs):
        patches = Variable(imgs)
        labels = Variable(labels)

        if torch.cuda.is_available():
            patches = patches.cuda()
            labels = labels.cuda()
                
        self.optimizer.zero_grad()

        output = self.cnn(patches=patches, **kwargs)

        loss = self.__loss__(output, labels)
        loss.backward()
        self.optimizer.step()
    
    def __evaluate_training__(self):
        psnr = self.evaluate()
        self.__track_psnr__(psnr)

        is_best = False
        if self.best_psnr < psnr:
            self.best_psnr = psnr
            is_best = True
        
        return is_best


    def train(self):
        assert self.cnn

        with closing(ScenesDeepHDR(root=os.path.join(Constants.training_data_root, Constants.training_directory))) as scenes:
            with tqdm(total=TrainingConstants.num_iterations) as pbar:
                pbar.update(self.starting_iteration)
                iteration = self.starting_iteration
                while iteration < TrainingConstants.num_iterations:
                    scene_loader = torch.utils.data.DataLoader(scenes, shuffle=True)
                    for (scene_imgs, scene_labels, index) in scene_loader:
                        patches = PatchesDeepHDR(scene_imgs.squeeze(), scene_labels.squeeze())
                        patches_loader = torch.utils.data.DataLoader(patches, batch_size=20, shuffle=True)
                        for (imgs, labels) in patches_loader:                            
                            self.__train__(imgs,labels)
                            
                            if iteration % TrainingConstants.validation_frequency == 0:
                                is_best = self.__evaluate_training__()
                                self.__make_checkpoint__(iteration, is_best)
                                
                            iteration += 1
                            pbar.update()
    
    def evaluate_imgs(self, imgs, **kwargs):
        patches = Variable(imgs)

        if torch.cuda.is_available():
            patches = patches.cuda()
                
        self.optimizer.zero_grad()

        return self.cnn(patches=patches, **kwargs)
                            
    def evaluate(self):
        with closing(ScenesDeepHDR(root=os.path.join(Constants.training_data_root, Constants.test_directory))) as scenes:
            scene_loader = torch.utils.data.DataLoader(scenes)
            psnrs = []
            for (scene_imgs, scene_labels, _) in tqdm(scene_loader):
                patches = PatchesDeepHDR(scene_imgs.squeeze(), scene_labels.squeeze())
                patches_loader = torch.utils.data.DataLoader(patches, batch_size=20)
                for (imgs, labels) in patches_loader:
                    output = self.evaluate_imgs(imgs)

                    psnrs.append(psnr(output.data, labels.cuda()))

            average_psnr = sum(psnrs)/len(psnrs)

            print("validation psnr : {}".format(average_psnr))

        return average_psnr
            
    @abstractmethod
    def __build_model__(self):
        pass

    def __make_checkpoint__(self, iteration, is_best, additionnal_name=None, **kwargs):
        checkpoint_datetime = str(datetime.now())

        if additionnal_name:
            filename = 'checkpoint_{}_{}.pth'
            filename = filename.format(checkpoint_datetime, additionnal_name)
        else:
            filename = 'checkpoint_{}.pth'
            filename = filename.format(checkpoint_datetime)
        
        state = {
            'iteration': iteration,
            'state_dict': self.cnn.state_dict(),
            'best_psnr': self.best_psnr,
            'optimizer' : self.optimizer.state_dict(),
        }
        state.update(kwargs)
        
        torch.save(state, self.checkpoints_folder + filename)
        if is_best:
            shutil.copyfile(self.checkpoints_folder + filename, self.checkpoints_folder + 'model_best.pth')
    
    def __load_checkpoint__(self, checkpoint):
        from_checkpoint = False
        if checkpoint:
            if os.path.isfile(checkpoint):
                from_checkpoint = True
                print("loading checkpoint '{}'".format(checkpoint))
                checkpoint = torch.load(checkpoint)
                self.starting_iteration = checkpoint['iteration']
                self.best_psnr = checkpoint['best_psnr']

                # create new OrderedDict that does not contain `module.`
                '''
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                state_dict = checkpoint['state_dict']
                for k, v in state_dict.items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                # load params
                self.cnn.load_state_dict(new_state_dict)
                '''
                self.cnn.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.__load_addtional_params__(checkpoint)
                print("=> loaded checkpoint '{}' (iteration {})"
                    .format(checkpoint, checkpoint['iteration']))
            else:
                print("no checkpoint found at '{}'".format(checkpoint))
                print("starting with no checkpoint")
        
        return from_checkpoint

    def __load_addtional_params__(self, checkpoint):
        pass
    
    def __track_psnr__(self, psnr):
        with open(self.psnr_track_file, "a") as myfile:
            myfile.write("{:.14f}\n".format(psnr))

class DirectTrainerDeepHDR(TrainerDeepHDR):
    def __init__(self, checkpoint, checkpoints_folder="./checkpoints/"):
        super(DirectTrainerDeepHDR, self).__init__("Direct", checkpoint, checkpoints_folder)
    
    def __build_model__(self):
        return DirectDeepHDR()

class WeTrainerDeepHDR(TrainerDeepHDR):
    def __init__(self, checkpoint, checkpoints_folder="./checkpoints/"):
        super(WeTrainerDeepHDR, self).__init__("WE", checkpoint, checkpoints_folder)
    
    def __build_model__(self):
        return WeDeepHDR()

class WieTrainerDeepHDR(TrainerDeepHDR):
    def __init__(self, checkpoint, checkpoints_folder="./checkpoints/", phase=1):
        super(WieTrainerDeepHDR, self).__init__("WIE", checkpoint, checkpoints_folder)
        self.scenes = sorted(list_all_folders(os.path.join(Constants.scenes_root, Constants.training_directory)))

        if phase==2:
            self.cnn.set_phase_2()
    
    def __build_model__(self):
        return WieDeepHDR()

    def read_exposure(self, is_training, scene):
        if is_training:
            folder = Constants.training_directory
        else:
            folder = Constants.test_directory

        return ReadExpoTimes(os.path.join(Constants.scenes_root, folder, self.scenes[scene], "exposure.txt"))

    def train_phase(self, dataset):
        with closing(dataset(os.path.join(Constants.training_data_root, Constants.training_directory))) as scenes:
            with tqdm(total=TrainingConstants.num_iterations) as pbar:
                pbar.update(self.starting_iteration)
                iteration = self.starting_iteration
                while iteration < TrainingConstants.num_iterations:
                    scene_loader = torch.utils.data.DataLoader(scenes, shuffle=True)
                    for (scene_imgs, scene_labels, index) in scene_loader:
                        patches = PatchesDeepHDR(scene_imgs.squeeze(), scene_labels.squeeze())
                        patches_loader = torch.utils.data.DataLoader(patches, batch_size=20, shuffle=True)
                        expos = self.read_exposure(True, int(index[0]))
                        for (imgs, labels) in patches_loader:
                            self.__train__(imgs, labels, expos=expos)

                            if iteration % TrainingConstants.validation_frequency == 0:
                                is_best = self.__evaluate_training__()
                                self.__make_checkpoint__(iteration, is_best, additionnal_name=dataset.__name__, phase=self.cnn.get_phase())

                            iteration += 1
                            pbar.update()
    
    def train(self):
        if self.cnn.get_phase() == 1:
            self.train_phase(RefinerScenesDeepHDR)
            self.best_psnr = 0
        elif self.cnn.get_phase() == 2:
            self.cnn.set_phase_2()
            self.train_phase(ScenesDeepHDR)
    
    def evaluate(self):
        if self.cnn.get_phase() == 1:
            dataset = RefinerScenesDeepHDR
        elif self.cnn.get_phase() == 2:
            dataset = ScenesDeepHDR

        psnrs = []   
        with closing(dataset(os.path.join(Constants.training_data_root, Constants.test_directory))) as scenes:
            scene_loader = torch.utils.data.DataLoader(scenes, shuffle=True)
            for (scene_imgs, scene_labels, index) in tqdm(scene_loader):
                patches = PatchesDeepHDR(scene_imgs.squeeze(), scene_labels.squeeze())
                patches_loader = torch.utils.data.DataLoader(patches, batch_size=20, shuffle=True)
                expos = self.read_exposure(False, int(index[0]))
                for (imgs, labels) in patches_loader:
                    output = self.evaluate_imgs(imgs, expos=expos)
                    psnrs.append(psnr(output.data, labels.cuda()))

        average_psnr = sum(psnrs)/len(psnrs)

        print("validation psnr : {}".format(average_psnr))
        
        return average_psnr

    def __load_addtional_params__(self, checkpoint):
        phase = checkpoint['phase']
        if phase == 2:
            self.cnn.set_phase_2()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Training a deep learning model to generate HDR images.',
    )
    
    parser.add_argument('-v', '--validate', action='store_true', dest='is_evaluate', default=False)

    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument('-b', '--best', dest="best_checkpoint", action='store_true', default=False)
    checkpoint_group.add_argument('-l', '--lastcheckpoint', dest="last_checkpoint", action='store_true', default=False)
    checkpoint_group.add_argument('-c', '--checkpoint', dest="checkpoint", type=str, default=None)

    parser.add_argument('-f', '--checkpointsfolder', dest="checkpoints_folder", type=str, default='./checkpoints/')

    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--Direct', dest='model', action='store_const', const='Direct')
    model_group.add_argument('--WE', dest='model', action='store_const', const='WE')
    model_group.add_argument('--WIE', dest='model', action='store_const', const='WIE')

    args = parser.parse_args()
    
    if args.best_checkpoint:
        checkpoint = os.path.join(args.checkpoints_folder, args.model, 'model_best.pth')
    elif args.last_checkpoint:
        list_of_files = glob.glob(os.path.join(args.checkpoints_folder, args.model, 'checkpoint_*.pth'))
        checkpoint = max(list_of_files, key=os.path.getctime)
    elif args.checkpoint:
        checkpoint = args.checkpoint
    else:
        checkpoint = None

    if args.model == 'Direct':
        trainer = DirectTrainerDeepHDR(checkpoint, args.checkpoints_folder)
    elif args.model == 'WE':
        trainer = WeTrainerDeepHDR(checkpoint, args.checkpoints_folder)
    elif args.model == 'WIE':
        trainer = WieTrainerDeepHDR(checkpoint, args.checkpoints_folder, phase=2)

    if args.is_evaluate:
        trainer.evaluate()
    else:
        trainer.train()