import torch
import torch.nn as nn
import torch.nn.functional as F


L2 = 1.0e-5


class conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, strides, padding, dilation_rate):
        super().__init__()
        self.conv2d = nn.Conv2D(in_channels, out_channels, kernel_size, strides, padding, dilation_rate)

    def forward(self, x):
        return self.conv2d(x)


class conv2d_bn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, strides, padding, dilation_rate):
        super().__init__()
        self.conv2d_bn = nn.Sequential(
            nn.Conv2D(in_channels, out_channels, kernel_size, strides, padding, dilation_rate),
            nn.BatchNormalization(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv2d_bn


class avg_pool(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size):
        super().__init__()
        self.avg_pool = nn.Sequential(
            nn.AvgPool2D(pool_size=pool_size),
            nn.Conv2D(in_channels, out_channels, 1, 1),
        )
    
    def forward(self, x):
        return self.avg_pool


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate):
        super().__init__()
        self.basic_block = nn.Sequential(
            self.conv1 = conv2d_bn(in_channels, out_channels, 3, 1, dilation_rate=dilation_rate),
            self.conv2 = conv2d_bn(in_channels, out_channels, 3, 1, dilation_rate=dilation_rate)
        )

    def forward(self, x):
        return self.basic_block


class make_blocks(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate, num):
        super().__init__()
        layers = []

    for i in range(num):
        layers.append(BasicBlock(in_channels, out_channels, dilation_rate))

    self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        return self.blocks


class FeatureExtraction(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv0_1 = conv2d_bn(in_channels, out_channels, 5, 2, 'same', 1)
        self.conv0_2 = conv2d_bn(2 * in_channels, 2* out_channels, 5, 2, 'same', 1)

        self.conv1_0 = make_blocks(2 * in_channels, 2* out_channels, 1, 4)
        self.conv1_1 = make_blocks(2 * in_channels, 2* out_channels, 2, 2)
        self.conv1_2 = make_blocks(2 * in_channels, 2* out_channels, 4, 2)
        self.conv1_3 = make_blocks(2 * in_channels, 2* out_channels, 1, 2)

        self.branch0 = avg_pool(in_channels, out_channels, 1)
        self.branch1 = avg_pool(in_channels, out_channels, 2)
        self.branch2 = avg_pool(in_channels, out_channels, 4)

    def forward(self, x):
        x = self.conv0_1(x)
        x = self.conv0_2(x)

        x = self.conv1_0(x)
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)

        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        return [x0, x1, x2]
