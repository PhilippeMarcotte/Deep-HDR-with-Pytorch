import torch.nn as nn
from ImageUtilities import extract_patches_from_image
import numpy as np
from torch.autograd import Variable
import torch

class DeepHDRCNN(nn.Module):
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

    def compute_HDR_image(self, x):
        return x

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        hdr_img = self.compute_HDR_image(out)
        return hdr_img#tone_map(hdr_img)

    def loss(self, result, target):
        assert result.size() == target.size()
        diff = result.sub(target)
        sqr_diff = diff ** 2
        return sqr_diff.sum(1)

    def train(self, data, label):
        print("HEYO")
        


class DirectDeepHDR(DeepHDRCNN):
    def __init__(self):
        super(DirectDeepHDR, self).__init__(3)

class WeDeepHDR(DeepHDRCNN):
    def __init__(self):
        super(WeDeepHDR, self).__init__(9)
    
    def compute_HDR_image(self, x):
        # return HDR image from weighted mean
        w1 = x[:,0:2,:,:]
        w2 = x[:,3:5,:,:]
        w3 = x[:,6:8,:,:]
        raise NotImplementedError("")        

class WieDeepHDR(DeepHDRCNN):
    def __init__(self):
        super(WieDeepHDR, self).__init__(18)

    def compute_HDR_image(self, x):
        # compute refined aligned HDR images
        # return HDR image from weighted mean        
        raise NotImplementedError("")

patch = Variable(torch.ones(1,18,40,40))

patch = patch.cuda()

cnn = DirectDeepHDR()

cnn = cnn.cuda()

output = cnn(patch)