import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F

from omegaconf import OmegaConf
from tqdm import tqdm
import wandb

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)
batch_size = 128
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5,0.5)])
train_dataset = datasets.CIFAR10('~/data/cifar10/', download=False, train=True, transform=transform)
test_dataset = datasets.CIFAR10('~/data/cifar10/', download=False, train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=1,padding = 0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 32, kernel_size=3, stride=1,padding = 0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(1152, 10),
            # torch.nn.Linear(64, 10),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)  
        x = self.conv2(x)  
        x = x.view(batch_size, -1)  
        x = self.fc(x)
        return x  

    
def eval(test_loader):
    correct = 0
    total = 0
    with torch.no_grad():  
        for data in test_loader:
            inputs, target = data
            total += inputs.shape[0]
            inputs, target = inputs.to(DEVICE), target.to(DEVICE)
    
            outputs = model(inputs)
    
            _, predicted = torch.max(outputs.data, dim=1)
            correct += (predicted == target).sum().cpu().detach().item()
            break
    return correct / total


def test(test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, target = data
            total += inputs.shape[0]
            inputs, target = inputs.to(DEVICE), target.to(DEVICE)

            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, dim=1)
            correct += (predicted == target).sum().cpu().detach().item()
    return correct / total

def train(acc_list):
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)

        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, dim=1)
        correct = ((predicted == target).sum()/ inputs.shape[0]).cpu().detach().item()
        acc_list.append(correct)
        if len(acc_list)%100 == 0:
            print(len(acc_list),"train:",correct," test:",eval(test_loader))
        
    return acc_list 

for i in range(1):
    learning_rate = 1e-3
    model = CNN().to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3) 
    num_epochs = 10

    acc_list = []

    for epoch in range(num_epochs):
        acc_list = train(acc_list)

    # np.savetxt('plot/data/figure1_3_bp_2l/'+str(i)+'.csv',acc_list[0:4000])