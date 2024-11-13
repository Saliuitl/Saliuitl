###########################################################################################
# Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py #
# Mainly changed the model forward() function                                             #
###########################################################################################

import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms

#from torchvision.models import resnet50, ResNet50_Weights

import numpy
import torch

X = numpy.random.uniform(-10, 10, 70).reshape(1, 7, -1)
# Y = np.random.randint(0, 9, 10).reshape(1, 1, -1)

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


class AtkDetCNNRawN(nn.Module):

    def __init__(self):
        super(AtkDetCNNRawN, self).__init__()
        self.conv1=torch.nn.Conv1d(in_channels=4, out_channels=16, kernel_size=2, stride=1, groups=4)
        self.avgpool1 = nn.AdaptiveAvgPool1d(16)
        self.norm1=nn.BatchNorm1d(16)
        self.conv2=torch.nn.Conv1d(in_channels=16, out_channels=16, kernel_size=2, stride=1, groups=4)
        self.avgpool2 = nn.AdaptiveAvgPool1d(16)
        self.norm2=nn.BatchNorm1d(16)
        """
        self.conv3=torch.nn.Conv1d(in_channels=4, out_channels=4, kernel_size=2, stride=1, groups=4)
        self.avgpool3 = nn.AdaptiveAvgPool1d(4)
        self.norm3=nn.BatchNorm1d(4)
        """
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid=nn.Sigmoid()
        self.linear = nn.Linear(16*16, 16*16)
        self.fc = nn.Linear(16*16, 1)
        self.flatten=torch.flatten
        #self.conv3=torch.nn.Conv1d(in_channels=20, out_channels=128, kernel_size=5, stride=2)


    def forward(self, x):
        x = self.conv1(x)
        x = self.avgpool1(x)
        x = self.relu(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.avgpool2(x)
        x = self.relu(x)
        x = self.norm2(x)
        """
        x = self.conv3(x)
        x = self.avgpool3(x)
        x = self.relu(x)
        x = self.norm3(x)
        """
        x = self.flatten(x, start_dim=1)
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear(x)
        x = self.relu(x)
        x = self.fc(x)
        #print(x.shape)
        x = self.sigmoid(x)
        #print(x.shape)

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


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


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
        model = AtkDetCNNRawLegatto
    if pretrained and path is not None:
        model.load_state_dict(path)
    return model

def mlp(pretrained=False, path=None, in_size=1):
    model = AtkDetMLP(in_size=in_size)
    if pretrained and path is not None:
        model.load_state_dict(path)
    return model

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
        #self.linear = nn.Linear(4*32, 64*32)
        self.linear1 = nn.Linear(12*12, 12*12*4)
        self.linear2 = nn.Linear(12*12*4, 12*12*4)
        self.fc = nn.Linear(12*12*4, 1)
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
        #print(x.shape)
        x = self.linear2(x)
        x = self.relu(x)
        #print(x.shape)

        x = self.fc(x)
        #print(x.shape)
        x = self.sigmoid(x)
        #print(x.shape)

        return x

class AtkDetCNNRawnnNEW(nn.Module):

    def __init__(self, in_feats=4):
        super(AtkDetCNNRaw, self).__init__()
        """
        self.conv1=torch.nn.Conv1d(in_channels=1, out_channels=4, kernel_size=2, stride=1)
        self.avgpool1 = nn.AdaptiveAvgPool1d(4)
        self.norm1=nn.BatchNorm1d(4)
        self.conv2=torch.nn.Conv1d(in_channels=4, out_channels=4, kernel_size=2, stride=1)
        self.avgpool2 = nn.AdaptiveAvgPool1d(4)
        self.norm2=nn.BatchNorm1d(4)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid=nn.Sigmoid()
        #self.linear = nn.Linear(4*32, 64*32)
        self.linear1 = nn.Linear(4*4, 4*4*4)
        self.linear2 = nn.Linear(4*4*4, 4*4*4)
        self.embedder = nn.Linear(4*4*4, 4*4)
        """
        #"""
        self.in_feats=in_feats
        self.conv11=torch.nn.Conv1d(in_channels=1, out_channels=12, kernel_size=2, stride=1)
        self.avgpool11 = nn.AdaptiveAvgPool1d(12)
        self.norm11=nn.BatchNorm1d(12)
        self.conv12=torch.nn.Conv1d(in_channels=12, out_channels=12, kernel_size=2, stride=1)
        self.avgpool12 = nn.AdaptiveAvgPool1d(12)
        self.norm12=nn.BatchNorm1d(12)
        self.linear11 = nn.Linear(12*12, 12*12*4)
        self.linear12 = nn.Linear(12*12*4, 12*12*4)
        self.embedder1 = nn.Linear(12*12*4, 4*4)
        self.fc1=nn.Linear(4*4, 1)
        if self.in_feats>=2:
            self.conv21=torch.nn.Conv1d(in_channels=1, out_channels=12, kernel_size=2, stride=1)
            self.avgpool21 = nn.AdaptiveAvgPool1d(12)
            self.norm21=nn.BatchNorm1d(12)
            self.conv22=torch.nn.Conv1d(in_channels=12, out_channels=12, kernel_size=2, stride=1)
            self.avgpool22 = nn.AdaptiveAvgPool1d(12)
            self.norm22=nn.BatchNorm1d(12)
            self.linear21 = nn.Linear(12*12, 12*12*4)
            self.linear22 = nn.Linear(12*12*4, 12*12*4)
            self.embedder2 = nn.Linear(12*12*4, 4*4)
            self.fc2=nn.Linear(4*4, 1)
        if self.in_feats>=3:
            self.conv31=torch.nn.Conv1d(in_channels=1, out_channels=12, kernel_size=2, stride=1)
            self.avgpool31 = nn.AdaptiveAvgPool1d(12)
            self.norm31=nn.BatchNorm1d(12)
            self.conv32=torch.nn.Conv1d(in_channels=12, out_channels=12, kernel_size=2, stride=1)
            self.avgpool32 = nn.AdaptiveAvgPool1d(12)
            self.norm32=nn.BatchNorm1d(12)
            self.linear31 = nn.Linear(12*12, 12*12*4)
            self.linear32 = nn.Linear(12*12*4, 12*12*4)
            self.embedder3 = nn.Linear(12*12*4, 4*4)
            self.fc3=nn.Linear(4*4, 1)
        if self.in_feats>=4:
            self.conv41=torch.nn.Conv1d(in_channels=1, out_channels=12, kernel_size=2, stride=1)
            self.avgpool41 = nn.AdaptiveAvgPool1d(12)
            self.norm41=nn.BatchNorm1d(12)
            self.conv42=torch.nn.Conv1d(in_channels=12, out_channels=12, kernel_size=2, stride=1)
            self.avgpool42 = nn.AdaptiveAvgPool1d(12)
            self.norm42=nn.BatchNorm1d(12)
            self.linear41 = nn.Linear(12*12, 12*12*4)
            self.linear42 = nn.Linear(12*12*4, 12*12*4)
            self.embedder4 = nn.Linear(12*12*4, 4*4)
            self.fc4=nn.Linear(4*4, 1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid=nn.Sigmoid()
        #self.linear = nn.Linear(4*32, 64*32)
        #"""
        self.concatenate=torch.cat
        self.fc = nn.Linear(4, 1)
        self.flatten=torch.flatten
        #self.conv3=torch.nn.Conv1d(in_channels=20, out_channels=128, kernel_size=5, stride=2)


    def forward(self, x):
        x1=x[:,0,:].unsqueeze(1)
        x1 = self.conv11(x1)
        x1 = self.avgpool11(x1)
        x1 = self.relu(x1)
        x1 = self.norm11(x1)
        x1 = self.conv12(x1)
        x1 = self.avgpool12(x1)
        x1 = self.relu(x1)
        x1 = self.norm12(x1)
        x1 = self.flatten(x1, start_dim=1)
        x1 = self.linear11(x1)
        x1 = self.relu(x1)
        x1 = self.linear12(x1)
        x1 = self.relu(x1)
        x1 = self.embedder1(x1)
        x1 = self.relu(x1)
        x1 = self.fc1(x1)
        x1 = self.sigmoid(x1)

        if self.in_feats>=2:
            x2=x[:,1,:].unsqueeze(1)
            x2 = self.conv21(x2)
            x2 = self.avgpool21(x2)
            x2 = self.relu(x2)
            x2 = self.norm21(x2)
            x2 = self.conv22(x2)
            x2 = self.avgpool22(x2)
            x2 = self.relu(x2)
            x2 = self.norm22(x2)
            x2 = self.flatten(x2, start_dim=1)
            x2 = self.linear21(x2)
            x2 = self.relu(x2)
            x2 = self.linear22(x2)
            x2 = self.relu(x2)
            x2 = self.embedder2(x2)
            x2 = self.relu(x2)
            x2 = self.fc2(x2)
            x2 = self.sigmoid(x2)
            if self.in_feats==2:
                x=self.concatenate([x1,x2],dim=1)
        if self.in_feats>=3:
            x3=x[:,2,:].unsqueeze(1)
            x3 = self.conv31(x3)
            x3 = self.avgpool31(x3)
            x3 = self.relu(x3)
            x3 = self.norm31(x3)
            x3 = self.conv32(x3)
            x3 = self.avgpool32(x3)
            x3 = self.relu(x3)
            x3 = self.norm32(x3)
            x3 = self.flatten(x3, start_dim=1)
            x3 = self.linear31(x3)
            x3 = self.relu(x3)
            x3 = self.linear32(x3)
            x3 = self.relu(x3)
            x3 = self.embedder3(x3)
            x3 = self.relu(x3)
            x3 = self.fc3(x3)
            x3 = self.sigmoid(x3)
            if self.in_feats==3:
                x=self.concatenate([x1,x2,x3,x4],dim=1)
        if self.in_feats>=4:
            x4=x[:,3,:].unsqueeze(1)
            x4 = self.conv41(x4)
            x4 = self.avgpool41(x4)
            x4 = self.relu(x4)
            x4 = self.norm41(x4)
            x4 = self.conv42(x4)
            x4 = self.avgpool42(x4)
            x4 = self.relu(x4)
            x4 = self.norm42(x4)
            x4 = self.flatten(x4, start_dim=1)
            x4 = self.linear41(x4)
            x4 = self.relu(x4)
            x4 = self.linear42(x4)
            x4 = self.relu(x4)
            x4 = self.embedder4(x4)
            x4 = self.relu(x4)
            x4 = self.fc4(x4)
            x4 = self.sigmoid(x4)
            x=self.concatenate([x1,x2,x3,x4],dim=1)
        x = self.fc(x)
        x = self.sigmoid(x)
        #print(x.shape)

        return x
