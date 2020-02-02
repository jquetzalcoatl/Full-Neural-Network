import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import mnist_loader
import random

class Net(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Net, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size,10)
    def forward(self,x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        v,i = x.max(2)
        # print(i)
        # return x        #.squeeze()
        return x    #i.squeeze()

def vectorized_result(target):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e0 = torch.zeros((1,1,10))
    e0[0,0,target[0]] = 1
    for j in range(1,len(target)):
        e = torch.zeros((1,1, 10))
        e[0,0,target[j]] = 1.0
        e0 = torch.cat((e0,e))
    return e0

def accuracy(model, test_loader, batch_size):
    correct = 0.0
    for data, target in test_loader:
        data, target = Variable(data.reshape(batch_size,1,784)), Variable(target)
        # target = torch.ones(batch_size, 1, 10, requires_grad=True)
        output = model(data)
        v, i = output.max(2)
        # print(i.squeeze().size(), target.size())
        pred = i.squeeze().data #output.data.max(1)[1]
        correct = correct + pred.eq(target.data).cpu().sum()
    return correct.data.numpy()/len(test_loader.dataset)


def FullNetMSE():
    model = Net(784, 30)
    batch_size = 100
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('./temp', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])), batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('./temp', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()])), batch_size=batch_size)

    optimizer = optim.SGD(model.parameters(), lr=20, momentum=0.)

    print("Accuracy before training ", accuracy(model, test_loader, batch_size))

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data.reshape(batch_size, 1, 784).to(torch.float), requires_grad=True), Variable(vectorized_result(target).to(torch.float), requires_grad=True)
        optimizer.zero_grad()
        result = model(data)
        print("Accuracy after epoch: ", batch_idx, accuracy(model, test_loader, batch_size))
        loss = nn.MSELoss()
        output = loss(result.to(torch.float), target.to(torch.float))
        output.backward()
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)
        optimizer.step()
    print("Accuracy after training: ", accuracy(model, test_loader, batch_size))
    # Accuracy before training: 0.1005 # Accuracy after training: 0.8738

def FullNetCE():
    model = Net(784, 30)
    batch_size = 100
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('./temp', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])), batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('./temp', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()])), batch_size=batch_size)

    optimizer = optim.SGD(model.parameters(), lr=20, momentum=0.)

    print("Accuracy before training ", accuracy(model, test_loader, batch_size))

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data.reshape(batch_size, 1, 784).to(torch.float), requires_grad=True), Variable(target.long())
        optimizer.zero_grad()
        result = model(data)
        print("Accuracy after epoch: ", batch_idx, accuracy(model, test_loader, batch_size))
        loss = nn.CrossEntropyLoss()
        output = loss(result.squeeze(), target.squeeze())
        output.backward()
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)
        optimizer.step()
    print("Accuracy after training: ", accuracy(model, test_loader, batch_size))
    
FullNetCE()
    

####################################
