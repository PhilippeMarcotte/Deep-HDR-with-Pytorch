import torch.nn as nn
from ModelUtilities import weighted_average
from ModelUtilities import LDR_to_HDR
from ModelUtilities import l2_distance
import numpy as np
from torch.autograd import Variable
import torch
from math import log
import Constants
from torchvision import transforms

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
        return x

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        hdr_img = self.other_forward_steps(out)
        return self.tone_map(hdr_img)

class DirectDeepHDR(DeepHDRModel):
    def __init__(self):
        super(DirectDeepHDR, self).__init__(3)

class WeDeepHDR(DeepHDRModel):
    def __init__(self, images):
        super(WeDeepHDR, self).__init__(9)
    
    def other_forward_steps(self, weights):
        return weighted_average(weights, self.images[:, 9:18], 3)

class WieDeepHDRRefiner(DeepHDRModel)
    def __init__(self):
        super(WieDeepHDRRefiner, self).__init__(9)

class WieDeepHDR(DeepHDRModel):
    def __init__(self, expo, gamma):
        super(WieDeepHDR, self).__init__()

    def other_forward_steps(self, x, ):
        
        # return HDR image from weighted mean
        raise NotImplementedError("")

    def refined_LDR_images(self, x):
        x = x.clamp(0,1)
        low_expo_HDR = LDR_to_HDR(x[:, 0:3])
        med_expo_HDR = LDR_to_HDR(x[:, 3:6])
        hig_expo_HDR = LDR_to_HDR(x[:, 6:9])

        HDR_imgages = [low_expo_HDR, med_expo_HDR, hig_expo_HDR]
        return torch.cat(HDR_imgages, 1)

def train_DirectDeepHDR(patches, labels):
        cnn = DirectDeepHDR()
        cnn = cnn.cuda()

        optimizer = torch.optim.Adam(cnn.parameters(), Constants.learning_rate)
        start_batch_index = 0
        end_batch_index = Constants.batch_size
        for i in range(Constants.num_iterations):
            batch_patches = Variable(patches[start_batch_index:end_batch_index]).cuda()
            batch_labels = Variable(labels[start_batch_index:end_batch_index]).cuda()

            start_batch_index = end_batch_index
            end_batch_index = (end_batch_index + Constants.batch_size - 1) % patches.size()[0] + 1

            optimizer.zero_grad()

            output = cnn(batch_patches)

            loss = l2_distance(output, batch_labels)

            print(loss.data[0])

            loss.backward()

            optimizer.step()
        
patch = torch.arange(0,40*18*40*40).view(40,18,40,40)

label = torch.arange(0,40*3*28*28).view(40,3,28,28)
label = label.clamp(0,1)

train_DirectDeepHDR(patch, label)