import torch.nn as nn
from ModelUtilities import *
import numpy as np
from torch.autograd import Variable
import torch
import ModelsConstants
import torchvision.datasets as datasets
from PIL import Image
from DeepHDRDatasets import *

class DeepHDRModel(nn.Module):
    def __init__(self, out_channels = 3, use_xavier_init_uniformally = True):
        super(DeepHDRModel, self).__init__()
        self.layers = []

        self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels  = 18,
                          out_channels = 100,
                          kernel_size = (7,7)),
                nn.ReLU()
            )
            
        self.layer2 = nn.Sequential(
                nn.Conv2d(in_channels  = 100,
                          out_channels = 100,
                          kernel_size = (5,5)),
                nn.ReLU()
            )

        self.layer3 = nn.Sequential(
                nn.Conv2d(in_channels  = 100,
                          out_channels = 50,
                          kernel_size = (3,3)),
                nn.ReLU()
            )

        self.layer4 = nn.Sequential(
                nn.Conv2d(in_channels  = 50,
                          out_channels = out_channels,
                          kernel_size = (1,1)),
                nn.Sigmoid()
            )
        
        if use_xavier_init_uniformally:
            init_weight = nn.init.xavier_uniform
        else:
            init_weight = nn.init.xavier_normal

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init_weight(module.weight)

    def other_forward_steps(self, x):
        return tone_map(hdr_img)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        imgs = self.other_forward_steps(out)
        return imgs

class DirectDeepHDR(DeepHDRModel):
    def __init__(self):
        super(DirectDeepHDR, self).__init__(3)
    
    def other_forward_steps(self, x):
        return x

class WeDeepHDR(DeepHDRModel):
    def __init__(self, images):
        super(WeDeepHDR, self).__init__(9)
    
    def other_forward_steps(self, weights):
        return weighted_average(weights, self.images[:, 9:18], ModelsConstants.num_channels)

class WieDeepHDRRefiner(DeepHDRModel):
    def __init__(self):
        super(WieDeepHDRRefiner, self).__init__(9)

class WieDeepHDR(DeepHDRModel):
    def __init__(self, expo, gamma):
        super(WieDeepHDR, self).__init__()
    
    def phase1_foward(self, x):
        return x

    def setPhase1():
        self.phase_forward = self.phase1_foward
    
    def phase2_foward(self, x):
        return weighted_average(weights, self.images[:, 9:18], ModelsConstants.num_channels)

    def setPhase2():
        self.phase_forward = self.phase2_foward

    def other_forward_steps(self, x, ):
        return self.phase_forward(x)

    def refined_LDR_images(self, x):
        x = x.clamp(0,1)
        low_expo_HDR = LDR_to_HDR(x[:, 0:3])
        med_expo_HDR = LDR_to_HDR(x[:, 3:6])
        hig_expo_HDR = LDR_to_HDR(x[:, 6:9])

        HDR_imgages = [low_expo_HDR, med_expo_HDR, hig_expo_HDR]
        return torch.cat(HDR_imgages, 1)
