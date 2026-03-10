# -*- coding: utf-8 -*-
import torch


class Decoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.cfg = cfg

        # Layer Definition
        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(256),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU()
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU()
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 32, kernel_size=1, bias=cfg.NETWORK.TCONV_USE_BIAS),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 16, kernel_size=1, bias=cfg.NETWORK.TCONV_USE_BIAS),
            torch.nn.BatchNorm3d(16),
            torch.nn.ReLU()
        )
        self.layer6 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(16, 8, kernel_size=1, bias=cfg.NETWORK.TCONV_USE_BIAS),
            torch.nn.BatchNorm3d(8),
            torch.nn.ReLU()
        )
        self.layer7 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(8, 1, kernel_size=1, bias=cfg.NETWORK.TCONV_USE_BIAS),
            torch.nn.Sigmoid()
        )

    def forward(self, volume_features):
        gen_volume = volume_features.view(-1, 512, 4, 4, 4)
        # print(gen_volume.size())   # torch.Size([batch_size, 512, 4, 4, 4])
        gen_volume = self.layer1(gen_volume)
        # print(gen_volume.size())   # torch.Size([batch_size, 256, 8, 8, 8])
        gen_volume = self.layer2(gen_volume)
        # print(gen_volume.size())   # torch.Size([batch_size, 128, 16, 16, 16])
        gen_volume = self.layer3(gen_volume)
        # print(gen_volume.size())   # torch.Size([batch_size, 64, 32, 32, 32])
        gen_volume = self.layer4(gen_volume)
        # print(gen_volume.size())   # torch.Size([batch_size, 32, 32, 32, 32])
        gen_volume = self.layer5(gen_volume)
        # print(gen_volume.size())   # torch.Size([batch_size, 16, 32, 32, 32])
        raw_feature = gen_volume = self.layer6(gen_volume)
        # print(gen_volume.size())   # torch.Size([batch_size, 8, 32, 32, 32])
        gen_volume = self.layer7(gen_volume)
        # print(gen_volume.size())   # torch.Size([batch_size, 1, 32, 32, 32])
        raw_feature = torch.cat((raw_feature, gen_volume), dim=1)
        # print(raw_feature.size())   # torch.Size([batch_size, 9, 32, 32, 32])
        return raw_feature, gen_volume
