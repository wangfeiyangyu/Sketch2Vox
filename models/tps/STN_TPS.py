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

        bias = target_control_points.view(-1)
        self.cnn.fc2.bias.data.copy_(bias)
        self.cnn.fc2.weight.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)
        points = self.cnn(x)
        return points.view(batch_size, -1, 2)

class STN_net(nn.Module):

    def __init__(self, cfg):
        super(STN_net, self).__init__()
        self.cfg = cfg

        r1 = self.cfg.STN.SPAN_RANGE
        r2 = self.cfg.STN.SPAN_RANGE
        assert r1 < 1 and r2 < 1 # if >= 1, arctanh will cause error in BoundedGridLocNet
        target_control_points = torch.Tensor(list(itertools.product(
            np.arange(-r1, r1 + 0.00001, 2.0 * r1 / (self.cfg.STN.GRID_SIZE - 1)),
            np.arange(-r2, r2 + 0.00001, 2.0 * r2 / (self.cfg.STN.GRID_SIZE - 1)),
        )))
        Y, X = target_control_points.split(1, dim=1)
        target_control_points = torch.cat([X, Y], dim=1)

        self.loc_net = UnBoundedGridLocNet(self.cfg.STN.GRID_SIZE, self.cfg.STN.GRID_SIZE, target_control_points)

        self.tps = TPSGridGen(self.cfg.CONST.IMG_H, self.cfg.CONST.IMG_W, target_control_points)


    def forward(self, x):
        batch_size = x.size(0)
        source_control_points = self.loc_net(x)
        source_coordinate = self.tps(source_control_points)
        grid = source_coordinate.view(batch_size, self.cfg.CONST.IMG_H, self.cfg.CONST.IMG_W, 2)
        transformed_x = F.grid_sample(x, grid)
        return transformed_x
