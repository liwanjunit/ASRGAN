import math
import torch
from torch import nn


class EDSR(nn.Module):
    def __init__(self, scale_factor, img_range=255., rgb_mean=(0.4488, 0.4371, 0.4040)):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(EDSR, self).__init__()

        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        if torch.cuda.is_available():
            self.mean = self.mean.cuda()

        self.conv_first = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.residual = ResidualBlock(64)
        res = [self.residual for _ in range(16)]
        self.res = nn.Sequential(*res)
        self.conv_after_res = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        upsample = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        self.upsample = nn.Sequential(*upsample)
        self.conv_last = nn.Conv2d(64, 3, kernel_size=9, padding=4)

    def forward(self, x):

        self.mean = self.mean.type_as(x)

        x = (x - self.mean) * self.img_range
        x = self.conv_first(x)
        res = self.conv_after_res(self.res(x))
        res += x
        upsample = self.upsample(res)
        x = self.conv_last(upsample)
        x = x / self.img_range + self.mean

        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(residual)
        residual = self.conv2(residual)

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x
