import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, out_channels = 3):
        super(CNN, self).__init__()
        self.layers = []

        self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels  = 18,
                          out_channels = 100,
                          kernel_size = (7,7)),
                nn.ReLU()
            ))
            
        self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels  = 100,
                          out_channels = 100,
                          kernel_size = (5,5)),
                nn.ReLU()
            ))

        self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels  = 100,
                          out_channels = 50,
                          kernel_size = (3,3)),
                nn.ReLU()
            ))

        self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels  = 50,
                          out_channels = out_channels,
                          kernel_size = (1,1)),
                nn.Sigmoid()
            ))
    
    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(x)
        
        return out            

class DirectCNN(CNN):
    def __init__(self):
        super(DirectCNN, self, 3).__init__()

class WeCNN(CNN):
    def __init__(self):
        super(DirectCNN, self, 9).__init__()

class WieCNN(CNN):
    def __init__(self):
        super(DirectCNN, self, 18).__init__()
