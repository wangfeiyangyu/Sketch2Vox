import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
# import models.pointnet.pointnet_utils as utils

class PointNet(torch.nn.Module):
    def __init__(self, channel=2):
        super(PointNet, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=channel, out_channels=64, kernel_size=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU()
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU()
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU()
        )

        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU()
        )

    def forward(self, points):
        B, N, C = points.size()
        points = points.transpose(2, 1)
        # print(points.size())    # torch.Size([batch_size, point_num, 2])
        local_fetures_1 = self.layer1(points)
        # print(features.size())    # torch.Size([batch_size, 64, point_num])
        local_fetures_2 = self.layer2(local_fetures_1)
        # print(features.size())    # torch.Size([batch_size, 64, point_num])
        local_fetures_3 = self.layer3(local_fetures_2)
        # print(features.size())    # torch.Size([batch_size, 128, point_num])
        global_fetures = self.layer4(local_fetures_3)
        # print(features.size())    # torch.Size([batch_size, 1024, point_num])
        global_fetures = torch.max(global_fetures, 2, keepdim=True)[0]
        # print(global_fetures.size())    # torch.Size([batch_size, 1024, 1])
        global_fetures = global_fetures.view(-1, 1024, 1).repeat(1, 1, N)
        # print(features.size())    # torch.Size([batch_size, 1024, point_num])
        fetures = torch.cat([local_fetures_2, global_fetures], 1)
        # print(features.size())    # torch.Size([batch_size, 1088, point_num])
        return fetures


'''
# classification
class PointNetEncoder(torch.nn.Module):
    def __init__(self, cfg):
        super(PointNetEncoder, self).__init__()
        self.cfg = cfg

        self.feat = PointNet(channel=2)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(1024, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU()
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.Dropout(p=0.4),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU()
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Linear(256, self.cfg.CONST.FEATURE_DIM),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.feat(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
'''

# segmentation
class PointNetEncoder(torch.nn.Module):
    def __init__(self, cfg):
        super(PointNetEncoder, self).__init__()
        self.cfg = cfg

        self.feat = PointNet(channel=2)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv1d(1088, 512, 1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU()
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv1d(512, 256, 1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU()
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv1d(256, 64, 1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU()
        )

    def forward(self, points):
        # batch_size, point_num, channel = points.size()
        point_features = self.feat(points)
        # print(features.size())    # torch.Size([batch_size, 1088, point_num])
        point_features = self.layer1(point_features)
        # print(features.size())    # torch.Size([batch_size, 512, point_num])
        point_features = self.layer2(point_features)
        # print(features.size())    # torch.Size([batch_size, 256, point_num])
        point_features = self.layer3(point_features)
        # print(features.size())    # torch.Size([batch_size, 64, point_num])
        point_features = point_features.transpose(2, 1)
        # print(features.size())    # torch.Size([batch_size, point_num, 64])
        return point_features

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss


def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss