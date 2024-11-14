import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms

import numpy
import torch

def attack_detector(pretrained=False, path=None):
    model = AtkDetNet()
    if pretrained and path is not None:
        model.load_state_dict(path)
    return model

def cnn(pretrained=False, path=None):
    model = AtkDetCNN()
    if pretrained and path is not None:
        model.load_state_dict(path)
    return model

def cnn_raw(pretrained=False, path=None, leg=False, in_feats=4):
    if not leg:
        model = AtkDetCNNRaw(in_feats=in_feats)
    else:
        return None
    if pretrained and path is not None:
        model.load_state_dict(path)
    return model

def mlp(pretrained=False, path=None, in_size=1):
    model = AtkDetMLP(in_size=in_size)
    if pretrained and path is not None:
        model.load_state_dict(path)
    return model

"""
Default AD architecture described in the paper (in_feats is n_f)
"""
class AtkDetCNNRaw(nn.Module):

    def __init__(self, in_feats=4):
        super(AtkDetCNNRaw, self).__init__()
        self.conv1=torch.nn.Conv1d(in_channels=in_feats, out_channels=12, kernel_size=2, stride=1)
        self.avgpool1 = nn.AdaptiveAvgPool1d(12)
        self.norm1=nn.BatchNorm1d(12)
        self.conv2=torch.nn.Conv1d(in_channels=12, out_channels=12, kernel_size=2, stride=1)
        self.avgpool2 = nn.AdaptiveAvgPool1d(12)
        self.norm2=nn.BatchNorm1d(12)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid=nn.Sigmoid()
        self.linear1 = nn.Linear(12*12, 12*12*4)
        self.linear2 = nn.Linear(12*12*4, 12*12*4)
        self.fc = nn.Linear(12*12*4, 1)
        self.flatten=torch.flatten


    def forward(self, x):
        x = self.conv1(x)
        x = self.avgpool1(x)
        x = self.relu(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.avgpool2(x)
        x = self.relu(x)
        x = self.norm2(x)
        x = self.flatten(x, start_dim=1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.fc(x)
        x = self.sigmoid(x)

        return x

"""
Older experimental versions of AD
"""
class Simple1DCNN(torch.nn.Module):
    def __init__(self):
        super(Simple1DCNN, self).__init__()
        self.layer1 = torch.nn.Conv1d(in_channels=7, out_channels=20, kernel_size=5, stride=2)
        self.act1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Conv1d(in_channels=20, out_channels=10, kernel_size=1)
    def forward(self, x):
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)

        log_probs = torch.nn.functional.log_softmax(x, dim=1)

        return log_probs

class AtkDetNet(nn.Module):

    def __init__(self):
        super(AtkDetNet, self).__init__()
        self.conv1=torch.nn.Conv1d(in_channels=1, out_channels=4, kernel_size=2, stride=1)
        self.avgpool1 = nn.AdaptiveAvgPool1d(4)
        self.norm1=nn.BatchNorm1d(4)
        self.conv2=torch.nn.Conv1d(in_channels=4, out_channels=4, kernel_size=2, stride=1)
        self.avgpool2 = nn.AdaptiveAvgPool1d(4)
        self.norm2=nn.BatchNorm1d(4)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid=nn.Sigmoid()
        self.linear1 = nn.Linear(4*4, 64)
        self.linear2 = nn.Linear(64, 64)
        self.fc = nn.Linear(64, 1)
        self.flatten=torch.flatten
        #self.conv3=torch.nn.Conv1d(in_channels=20, out_channels=128, kernel_size=5, stride=2)


    def forward(self, x):
        #input(x.shape)
        x = self.conv1(x)
        #print(x.shape)
        x = self.avgpool1(x)
        #print(x.shape)
        x = self.relu(x)
        #print(x.shape)
        x = self.norm1(x)
        #print(x.shape)
        x = self.conv2(x)
        #print(x.shape)
        x = self.avgpool2(x)
        #print(x.shape)
        x = self.relu(x)
        #print(x.shape)
        x = self.norm2(x)
        #print(x.shape)
        x = self.flatten(x, start_dim=1)
        #print(x.shape)
        #x = self.linear(x)
        #print(x.shape)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.fc(x)
        #print(x.shape)
        x = self.sigmoid(x)
        #print(x.shape)

        return x

class AtkDetCNN(nn.Module):

    def __init__(self):
        super(AtkDetCNN, self).__init__()
        self.conv1=torch.nn.Conv1d(in_channels=3, out_channels=12, kernel_size=4, stride=1)
        self.avgpool1 = nn.AdaptiveAvgPool1d(12)
        self.norm1=nn.BatchNorm1d(12)
        self.conv2=torch.nn.Conv1d(in_channels=12, out_channels=12, kernel_size=2, stride=1)
        self.avgpool2 = nn.AdaptiveAvgPool1d(12)
        self.norm2=nn.BatchNorm1d(12)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid=nn.Sigmoid()
        #self.linear = nn.Linear(4*32, 64*32)
        self.fc = nn.Linear(12*12, 1)
        self.flatten=torch.flatten
        #self.conv3=torch.nn.Conv1d(in_channels=20, out_channels=128, kernel_size=5, stride=2)


    def forward(self, x):
        #input(x)
        x = self.conv1(x)
        #print(x.shape)
        x = self.avgpool1(x)
        #print(x.shape)
        x = self.relu(x)
        #print(x.shape)
        x = self.norm1(x)
        #print(x.shape)
        x = self.conv2(x)
        #print(x.shape)
        x = self.avgpool2(x)
        #print(x.shape)
        x = self.relu(x)
        #print(x.shape)
        x = self.norm2(x)
        #print(x)
        x = self.flatten(x, start_dim=1)
        #print(x)
        #x = self.linear(x)
        #print(x.shape)
        x = self.fc(x)
        #print(x)
        x = self.sigmoid(x)
        #input(x)

        return x


class AtkDetMLP(nn.Module):

    def __init__(self, in_size=1):
        super(AtkDetMLP, self).__init__()
        #self.conv1=torch.nn.Conv1d(in_channels=20, out_channels=32, kernel_size=1, stride=1)
        #self.avgpool1 = nn.AdaptiveAvgPool1d(32)
        self.norm1=nn.BatchNorm1d(128)
        #self.conv2=torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2, stride=1)
        #self.avgpool2 = nn.AdaptiveAvgPool1d(32)
        self.norm2=nn.BatchNorm1d(128)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid=nn.Sigmoid()
        self.linear = nn.Linear(20*in_size, 128)
        self.linear2 = nn.Linear(128,128)
        self.fc = nn.Linear(128, 1)
        self.flatten=torch.flatten
        #self.conv3=torch.nn.Conv1d(in_channels=20, out_channels=128, kernel_size=5, stride=2)


    def forward(self, x):
        #input(x.shape)
        x=self.flatten(x, start_dim=1)
        x = self.linear(x)
        #print(x.shape)
        #3x = self.avgpool1(x)
        #print(x.shape)
        x = self.relu(x)
        #print(x.shape)
        x = self.norm1(x)
        #print(x.shape)
        x = self.linear2(x)
        #print(x.shape)
        #x = self.avgpool2(x)
        #print(x.shape)
        x = self.relu(x)
        #print(x.shape)
        x = self.norm2(x)
        #print(x.shape)
        #x = self.flatten(x, start_dim=1)
        #print(x.shape)
        #x = self.linear(x)
        #print(x.shape)
        x = self.fc(x)
        #print(x.shape)
        x = self.sigmoid(x)
        #print(x.shape)

        return x
        return x
