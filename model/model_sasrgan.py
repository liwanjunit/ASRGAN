import math
import torch
from torch import nn
from attention import InterlacedSparseSelfAttention, Attention_D


class Generator_SASRGAN(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator_SASRGAN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = ResidualBlock(64)
        self.block8 = ResidualBlock(64)
        self.block9 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )
        block10 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block10.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.block10 = nn.Sequential(*block10)

    def forward(self, x):

        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block7)
        block9 = self.block9(block8)
        block10 = self.block10(block1 + block9)
        return (torch.tanh(block10) + 1) / 2


class Discriminator_SASRGAN(nn.Module):
    def __init__(self):
        super(Discriminator_SASRGAN, self).__init__()

        # self.attention = Attention(dim=512, P_h=8, P_w=8)
        self.attention = Attention_D(dim=256)

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            #
            # nn.Conv2d(256, 512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512),
            # nn.LeakyReLU(0.2),
            #
            # nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(512),
            # nn.LeakyReLU(0.2),
        )

        self.dense = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )


    def forward(self, x):
        batch_size = x.size(0)
        x1 = self.conv(x)
        x2 = self.attention(x1)
        out = self.dense(x1 + x2)
        return torch.sigmoid(out.view(batch_size))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.prelu = nn.PReLU()
        self.attention = InterlacedSparseSelfAttention(dim=64, P_h=8, P_w=8)

    def forward(self, x):

        x1 = self.prelu(self.conv1(x))
        # print(f'x.shape: {x.shape}')
        # print(f'x1.shape: {x1.shape}')
        x2 = self.prelu(self.conv2(x + x1))
        x3 = self.attention(x + x1 + x2)

        return x + x3 * 0.5
        # return x + x3


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
