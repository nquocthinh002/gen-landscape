import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=128, channels_img=3, features_g=64):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x z_dim x 1 x 1
            self._block(z_dim, features_g * 16, 4, 1, 0),  # 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),   # 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),   # 32x32
            self._block(features_g * 2, features_g, 4, 2, 1),       # 64x64
            self._block(features_g, features_g//2, 4, 2, 1),        # 128x128
            nn.ConvTranspose2d(features_g//2, channels_img, 4, 2, 1),  # 256x256
            nn.Tanh(), # output range [-1, 1]
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, channels_img=3, features_d=64):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x 3 x 256 x 256
            nn.Conv2d(channels_img, features_d, 4, 2, 1),  # 128x128
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d*2, 4, 2, 1), # 64x64
            self._block(features_d*2, features_d*4, 4, 2, 1), # 32x32
            self._block(features_d*4, features_d*8, 4, 2, 1), # 16x16
            self._block(features_d*8, features_d*16, 4, 2, 1), # 8x8
            self._block(features_d*16, features_d*32, 4, 2, 1), # 4x4
            nn.Conv2d(features_d*32, 1, 4, 1, 0),  # 1x1
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.net(x)
