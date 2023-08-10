import torch
import torch.nn as nn
from torchinfo import summary
import numpy as np
import matplotlib.pyplot as plt

'''
Notes

1) Wi2Vi uses 56x3x3x29 CSI, while we use 30x3x3x100

2) Video frames are aligned with the first packets of CSI

3) Wi2Vi video FPS = 30 -> 6, CSI rate = 100Hz

4) Wi2Vi train:test = 95:5

5) Wi2Vi lr=2e-3 and lower; epoch=1000; batch size=32

'''


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=padding),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU()
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out


class Interpolate(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x


class DropIn(nn.Module):
    def __init__(self, num_select):
        super(DropIn, self).__init__()
        self.num_select = num_select

    def forward(self, x):
        i = torch.randperm(x.shape[-1])[:self.num_select]
        return x[i]


class Wi2Vi(nn.Module):
    def __init__(self):
        super(Wi2Vi, self).__init__()

        self.Dropin = DropIn(17)
        self.Encoder = nn.Sequential(
            # 56x17x18?
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            # 56x15x64
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            # 26x7x128
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            # 12x3x256
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(),
            # 5x1x512
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(),
            # 5x1x512
        )

        self.Translator_A = nn.Sequential(
            nn.Linear(2560, 972),
            nn.LeakyReLU()
        )

        self.Translator_B = nn.Sequential(
            # 36x27
            nn.ReflectionPad2d(1),
            # 38x29
            nn.Conv2d(1, 32, kernel_size=7, stride=1, padding=0),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            # 32x23x32
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            # 16x12x64
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            # 8x6x128
        )

        self.Decoder = nn.Sequential(
            # 8x6x128
            nn.ReflectionPad2d(1),
            # 10x8x128
            ResidualBlock(128, 128, 0),
            # 8x6x128
            ResidualBlock(128, 128, 1),
            # 8x6x128
            ResidualBlock(128, 128, 1),
            # 8x6x128
            nn.functional.interpolate((16, 12)),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0),
            # 14x10x64
            nn.functional.interpolate((28, 20)),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0),
            # 26x18x32
            nn.functional.interpolate((52, 36)),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=0),
            # 50x34x16
            nn.functional.interpolate((100, 68)),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=0),
            # 98x66x8
            nn.functional.interpolate((196, 132)),
            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=0),
            # 194x130x4
            nn.functional.interpolate((388, 260)),
            nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=0),
            # 386x258x2
            nn.Conv2d(2, 1, kernel_size=5, stride=1, padding=0),
            nn.InstanceNorm2d(32),
            nn.ReLU()
            # 382x254x1
        )

    def forward(self, x):
        x = self.DropIn(x)
        x = self.Encoder(x)
        x = self.Translator_A(x.view(-1, 2560))
        x = self.Translator_B(x.view(-1, 1, 36, 27))
        x = self.Decoder(x)

        return x[..., 31:351, 7:247]
