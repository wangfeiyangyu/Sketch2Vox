# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>
#
# References:
# - https://github.com/shawnxu1318/MVCNN-Multi-View-Convolutional-Neural-Networks/blob/master/mvcnn.py

import torch
import torchvision.models
import utils.network_utils


class Encoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.cfg = cfg

        # Layer Definition
        vgg16_bn = torchvision.models.vgg16_bn(pretrained=True)
        self.vgg = torch.nn.Sequential(*list(vgg16_bn.features.children()))[:27]

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3),
            torch.nn.BatchNorm2d(512),
            torch.nn.ELU(),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3),
            torch.nn.BatchNorm2d(512),
            torch.nn.ELU(),
            torch.nn.MaxPool2d(kernel_size=3)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ELU()
        )
        self.FC1 = torch.nn.Sequential(
            torch.nn.Linear(256 * 8 * 8, self.cfg.CONST.Z_LENGTH),
            torch.nn.BatchNorm1d(self.cfg.CONST.Z_LENGTH),
            torch.nn.ELU()
        )
        self.FC2 = torch.nn.Sequential(
            torch.nn.Linear(256 * 8 * 8, self.cfg.CONST.Z_LENGTH),
            torch.nn.BatchNorm1d(self.cfg.CONST.Z_LENGTH),
            torch.nn.ELU()
        )

        # Don't update params in VGG16
        for param in vgg16_bn.parameters():
            param.requires_grad = False

    def forward(self, rendering_images):
        # print(rendering_images.size())  # torch.Size([batch_size, n_views, img_c, img_h, img_w])
        rendering_images = rendering_images.permute(1, 0, 2, 3, 4).contiguous()
        rendering_images = torch.split(rendering_images, 1, dim=0)
        # image_features = []
        img_Zmeans = []  # the mean value of imgVector
        img_ZlogVars = []  # the log variance of imgVector

        for img in rendering_images:
            features = self.vgg(img.squeeze(dim=0))
            # print(features.size())    # torch.Size([batch_size, 512, 28, 28])
            features = self.layer1(features)
            # print(features.size())    # torch.Size([batch_size, 512, 26, 26])
            features = self.layer2(features)
            # print(features.size())    # torch.Size([batch_size, 512, 24, 24])
            features = self.layer3(features)
            # print(features.size())    # torch.Size([batch_size, 256, 8, 8])
            features = features.reshape(-1, features.shape[-3]*features.shape[-2]*features.shape[-1])  # flatten
            z_mean = self.FC1(features)
            img_Zmeans.append(z_mean)
            z_log_var = self.FC2(features)
            img_ZlogVars.append(z_log_var)

        img_Zmeans = torch.stack(img_Zmeans).permute(1, 0, 2).contiguous()
        # print(img_Zmeans.size())  # torch.Size([batch_size, n_views, 512])
        img_ZlogVars = torch.stack(img_ZlogVars).permute(1, 0, 2).contiguous()
        # print(img_logVars.size())  # torch.Size([batch_size, n_views, 512])
        return img_Zmeans, img_ZlogVars

    def reparameterize(self, mu, var):
        if (self.cfg.CONST.get('WEIGHTS') is None) or self.cfg.TRAIN.RESUME_TRAIN:
            std = var.mul(0.5).exp_()
            eps = utils.network_utils.var_or_cuda((std.data.new(std.size()).normal_()))
            z = eps.mul(std).add_(mu)
            return z
        else:
            return mu
