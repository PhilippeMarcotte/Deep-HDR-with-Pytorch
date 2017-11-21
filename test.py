from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear = nn.Linear(3,2)
    
    def forward(self, x):
        return self.linear(x)

net = Net()

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

x = Variable(torch.randn(2, 2, 5, 3).cuda())
y = Variable(torch.randn(2, 2, 5, 2).cuda())

net = net.cuda()

for epoch in range(50):
    optimizer.zero_grad()

    outputs = net(x)

    loss= criterion(outputs, y)
    loss.backward()
    optimizer.step()

    print(loss.data[0])
    


'''
# Create tensors.
x = Variable(torch.randn(5, 3).cuda())
y = Variable(torch.randn(5, 2).cuda())

# Build a linear layer.
linear = nn.Linear(3, 2)
print ('w: ', linear.weight)
print ('b: ', linear.bias)

# Build Loss and Optimizer.
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

# Forward propagation.
pred = linear(x)

# Compute loss.
loss = criterion(pred, y)
print('loss: ', loss.data[0])

# Backpropagation.
loss.backward()

# Print out the gradients.
print ('dL/dw: ', linear.weight.grad) 
print ('dL/db: ', linear.bias.grad)

# 1-step Optimization (gradient descent).
optimizer.step()

# You can also do optimization at the low level as shown below.
# linear.weight.data.sub_(0.01 * linear.weight.grad.data)
# linear.bias.data.sub_(0.01 * linear.bias.grad.data)

# Print out the loss after optimization.
while True:
    pred = linear(x)
    loss = criterion(pred, y)
    print('loss after 1 step optimization: ', loss.data[0])
    loss.backward()
    optimizer.step()
'''
