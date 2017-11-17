import torch.nn as nn
from ImageUtilities import weighted_average
from ImageUtilities import LDR_to_HDR
import numpy as np
from torch.autograd import Variable
import torch

class DeepHDRModel(nn.Module):
    def __init__(self, out_channels = 3, use_xavier_init_uniformally = True):
        super(DeepHDRCNN, self).__init__()
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
        
        self.optimzer = torch.optim.Adam(self.parameters(), 0.0001)

        self.gamma = 2.2

    def other_forward_steps(self, x):
        return x

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        hdr_img = self.other_forward_steps(out)
        return hdr_img#tone_map(hdr_img)

    def loss(self, result, target):
        assert result.size() == target.size()
        return torch.sum((input - target) ** 2, dim=1)

    def train(self, images, expos, labels):
        self.images = images
        print("HEYO")

class DirectDeepHDR(DeepHDRModel):
    def __init__(self):
        super(DirectDeepHDR, self).__init__(3)

class WeDeepHDR(DeepHDRModel):
    def __init__(self):
        super(WeDeepHDR, self).__init__(9)
    
    def other_forward_steps(self, weights):
        return weighted_average(weights, self.images[:, 9:18], 3)

class WieDeepHDR(DeepHDRModel):
    def __init__(self):
        super(WieDeepHDR, self).__init__(9)

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

patch = Variable(torch.arange(0,9*40*40).view(1,9,40,40))

#print(patch[0,0:3,:,:])

zeros = Variable(torch.zeros(1,3,40,40))
ones  = Variable(torch.ones(1,3,40,40))
twos = Variable(2 * torch.ones(1,3,40,40))

mats = [zeros, ones, twos]

mat1 = torch.cat(mats, 1)
mat2 = torch.cat(mats, 1)

sum_mat = mat1 + mat2

print(sum_mat.size())
print((ones.matmul(ones) / twos)[0,0])
'''
patch = patch.cuda()

cnn = DirectDeepHDR()

cnn = cnn.cuda()

output = cnn(patch)
'''