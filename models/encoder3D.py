import torch


class Encoder3D(torch.nn.Module):
    def __init__(self, cfg):
        super(Encoder3D, self).__init__()
        self.cfg = cfg

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, 8, kernel_size=1, bias=cfg.NETWORK.TCONV_USE_BIAS),
            torch.nn.BatchNorm3d(8),
            torch.nn.LeakyReLU(self.cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(8, 32, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.LeakyReLU(self.cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(32, 128, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(self.cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv3d(128, 512, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(512),
            torch.nn.LeakyReLU(self.cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv3d(512, 2048, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(2048),
            torch.nn.LeakyReLU(self.cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer6 = torch.nn.Sequential(
            torch.nn.Linear(in_features=16384, out_features=4096),
            torch.nn.BatchNorm1d(4096),
            torch.nn.LeakyReLU(self.cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer7 = torch.nn.Sequential(
            torch.nn.Linear(in_features=4096, out_features=64),
            torch.nn.BatchNorm1d(64),
            # torch.nn.Sigmoid()
            torch.nn.ReLU()
        )


    def forward(self, volumes):
        # print(volumes.size())  # torch.Size([batch_size, n_vox, n_vox, n_vox])
        volumes_features = volumes.view(-1, 1, self.cfg.CONST.N_VOX, self.cfg.CONST.N_VOX, self.cfg.CONST.N_VOX)
        #print(volumes_features.size()) # torch.Size([batch_size, 1, 32, 32, 32])
        volumes_features = self.layer1(volumes_features)
        #print(volumes_features.size())  # torch.Size([batch_size, 8, 32, 32, 32])
        volumes_features = self.layer2(volumes_features)
        #print(volumes_features.size())  # torch.Size([batch_size, 32, 16, 16, 16])
        volumes_features = self.layer3(volumes_features)
        #print(volumes_features.size())  # torch.Size([batch_size, 128, 8, 8, 8])
        volumes_features = self.layer4(volumes_features)
        #print(volumes_features.size())  # torch.Size([batch_size, 512, 4, 4, 4])
        volumes_features = self.layer5(volumes_features)
        #print(volumes_features.size())  # torch.Size([batch_size, 2048, 2, 2, 2])
        volumes_vectors = torch.flatten(volumes_features, start_dim=1)
        #print(volumes_vectors.size())  # torch.Size([batch_size, 16384])
        volumes_vectors = self.layer6(volumes_vectors)
        #print(volumes_vectors.size())  # torch.Size([batch_size, 4096])
        volumes_vectors = self.layer7(volumes_vectors)
        #print(volumes_vectors.size())  # torch.Size([batch_size, 64])
        return volumes_vectors
