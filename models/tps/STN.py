# encoding: utf-8

import math
import torch
import itertools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.tps.tps_grid_gen import TPSGridGen

class CNN(nn.Module):
    def __init__(self, num_output):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=2)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(in_features=3380, out_features=50)
        self.fc2 = nn.Linear(50, num_output)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 3380)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

class BoundedGridLocNet(nn.Module):

    def __init__(self, grid_height, grid_width, target_control_points):
        super(BoundedGridLocNet, self).__init__()
        self.cnn = CNN(grid_height * grid_width * 2)

        bias = torch.from_numpy(np.arctanh(target_control_points.numpy()))
        bias = bias.view(-1)
        self.cnn.fc2.bias.data.copy_(bias)
        self.cnn.fc2.weight.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)
        points = F.tanh(self.cnn(x))
        return points.view(batch_size, -1, 2)

class UnBoundedGridLocNet(nn.Module):

    def __init__(self, grid_height, grid_width, target_control_points):
        super(UnBoundedGridLocNet, self).__init__()
        self.cnn = CNN(grid_height * grid_width * 2)



    def forward(self, x):
        batch_size = x.size(0)
        points = self.cnn(x)
        return points.view(batch_size, -1, 2)

class STN_net(nn.Module):

    def __init__(self, cfg):
        super(STN_net, self).__init__()
        self.cfg = cfg

        self.conv1 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=1)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(in_features=800, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=6)

        bias = torch.from_numpy(np.array([1, 0, 0, 0, 1, 0]))
        nn.init.constant_(self.fc2.weight, 0)
        self.fc2.bias.data.copy_(bias)

    def forward(self, imgs):
        batch_size = imgs.size(0)
        # print(imgs.size())    # torch.Size([batch_size, 512, 28, 28])
        x = F.relu(F.max_pool2d(self.conv1(imgs), 2))
        # print(x.size())    # torch.Size([batch_size, 128, 13, 13])
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # print(x.size())    # torch.Size([batch_size, 32, 5, 5])
        x = x.view(-1, 800)
        # print(x.size())    # torch.Size([batch_size, 800])
        x = F.relu(self.fc1(x))
        # print(x.size())    # torch.Size([batch_size, 128])
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        # print(x.size())    # torch.Size([batch_size, 6])
        theta = x.view(batch_size, 2, 3)

        grid = F.affine_grid(theta, imgs.size())
        transformed_imgs = F.grid_sample(imgs, grid)
        return transformed_imgs
