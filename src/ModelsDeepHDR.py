import torch.nn as nn
from ModelUtilities import *
import numpy as np
from torch.autograd import Variable
import torch
import Constants
import torchvision.datasets as datasets
from PIL import Image
from DatasetsDeepHDR import *
import imageio

class ModelDeepHDR(nn.Module):
    def __init__(self, out_channels = 3, use_xavier_init_uniformally = True):
        super(ModelDeepHDR, self).__init__()

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

    def post_convolution_steps(self, out, **inputs):
        tone_mapping = False
        if 'tone_mapping' in inputs: tone_mapping = inputs['tone_mapping']

        if tone_mapping:
            return range_compressor(out)
        else:
            return out

    def forward(self, **inputs):
        out = self.layer1(inputs['patches'])
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        imgs = self.post_convolution_steps(out, **inputs)
        return imgs

class DirectDeepHDR(ModelDeepHDR):
    def __init__(self):
        super(DirectDeepHDR, self).__init__(3)

class WeDeepHDR(ModelDeepHDR):
    def __init__(self):
        super(WeDeepHDR, self).__init__(9)
    
    def post_convolution_steps(self, weights, **inputs):
        imgs = crop_center(inputs['patches'][:, 9:18], Constants.cnn_crop_size)
        hdr_imgs = weighted_average(weights, imgs, Constants.num_channels)
        return super(WeDeepHDR, self).post_convolution_steps(hdr_imgs)

class WieDeepHDR(ModelDeepHDR):
    def __init__(self):
        super(WieDeepHDR, self).__init__(18)
        self.set_phase_1()

    def phase_1_steps(self, out, **inputs):
        return out[:, 0:9]

    def set_phase_1(self):
        self.steps = self.phase_1_steps
        self.phase = 1

    def phase_2_steps(self, out, **inputs):
        hdr_imgs = []

        for i in range(3):
            hdr_imgs.append(LDR_to_HDR(out[:, i * 3: (i+1) * 3], inputs['expos'][i], Constants.gamma))
        
        hdr_imgs = torch.cat(hdr_imgs, 1)
        return weighted_average(out[:, 9:18], hdr_imgs, Constants.num_channels)
    
    def set_phase_2(self):
        self.steps = self.phase_2_steps
        self.phase = 2
    
    def post_convolution_steps(self, out, **inputs):
        return self.steps(out, **inputs)
    
    def eval(self):
        self.steps = self.phase_2_steps
    
    def train(self):
        self.steps = self.phase_1_steps
    
    def get_phase(self):
        return self.phase
