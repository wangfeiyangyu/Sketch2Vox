# -*- coding: utf-8 -*-
import torch


class Decoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.cfg = cfg

        # Layer Definition
        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(in_channels=128, out_channels=32, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(in_channels=32, out_channels=8, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(8),
            torch.nn.ReLU()
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(in_channels=8, out_channels=1, kernel_size=1, bias=cfg.NETWORK.TCONV_USE_BIAS),
            torch.nn.Sigmoid()
        )

    def forward(self, volume_features):
        gen_volume = volume_features.view(-1, 256, 4, 4, 4)
        # print(gen_volume.size())   # torch.Size([batch_size, 256, 4, 4, 4])
        gen_volume = self.layer1(gen_volume)
        # print(gen_volume.size())   # torch.Size([batch_size, 128, 8, 8, 8])
        gen_volume = self.layer2(gen_volume)
        # print(gen_volume.size())   # torch.Size([batch_size, 32, 16, 16, 16])
        raw_feature = gen_volume = self.layer3(gen_volume)
        # print(gen_volume.size())   # torch.Size([batch_size, 8, 32, 32, 32])
        gen_volume = self.layer4(gen_volume)
        # print(gen_volume.size())   # torch.Size([batch_size, 1, 32, 32, 32])
        raw_feature = torch.cat((raw_feature, gen_volume), dim=1)
        # print(raw_feature.size())   # torch.Size([batch_size, 9, 32, 32, 32])
        return raw_feature, gen_volume
