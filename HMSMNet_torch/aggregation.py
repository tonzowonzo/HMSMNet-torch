import torch
import torch.nn as nn


L2 = 1.0e-5
alpha = 0.2


def conv3d(filters, kernel_size, strides, padding):
    return nn.Conv3d(filters, filters, kernel_size=kernel_size,
                     stride=strides, padding=padding,
                     bias=False)


def conv3d_bn(filters, kernel_size, strides, padding, activation):
    conv = nn.Conv3d(filters, filters, kernel_size=kernel_size,
                     stride=strides, padding=padding,
                     bias=False)
    bn = nn.BatchNorm3d(filters)
    leaky_relu = nn.LeakyReLU(alpha=alpha)

    if activation:
        return nn.Sequential(conv, bn, leaky_relu)
    else:
        return nn.Sequential(conv, bn)


def trans_conv3d_bn(filters, kernel_size, strides, padding, activation):
    conv = nn.ConvTranspose3d(filters, filters, kernel_size=kernel_size,
                              stride=strides, padding=padding,
                              bias=False)
    bn = nn.BatchNorm3d(filters)
    leaky_relu = nn.LeakyReLU(alpha=alpha)

    if activation:
        return nn.Sequential(conv, bn, leaky_relu)
    else:
        return nn.Sequential(conv, bn)


class Hourglass(nn.Module):
    def __init__(self, filters):
        super(Hourglass, self).__init__()

        self.conv1 = conv3d_bn(filters, 3, 1, 1, True)
        self.conv2 = conv3d_bn(filters, 3, 1, 1, True)
        self.conv3 = conv3d_bn(2 * filters, 3, 2, 1, True)
        self.conv4 = conv3d_bn(2 * filters, 3, 1, 1, True)
        self.conv5 = conv3d_bn(2 * filters, 3, 2, 1, True)
        self.conv6 = conv3d_bn(2 * filters, 3, 1, 1, True)
        self.conv7 = trans_conv3d_bn(2 * filters, 4, 2, 1, True)
        self.conv8 = trans_conv3d_bn(filters, 4, 2, 1, True)

    def forward(self, inputs):
        x1 = self.conv1(inputs)
        x1 = self.conv2(x1)
        x2 = self.conv3(x1)
        x2 = self.conv4(x2)
        x3 = self.conv5(x2)
        x3 = self.conv6(x3)
        x4 = self.conv7(x3)
        x4 += x2
        x5 = self.conv8(x4)
        x5 += x1

        return x5  # [N, C, D, H, W]


class FeatureFusion(nn.Module):
    def __init__(self, units):
        super(FeatureFusion, self).__init__()

        self.upsample = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=True)
        self.add1 = nn.Add()
        self.avg_pool3d = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Linear(units, units, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(units, units, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.multiply1 = nn.Multiply()
        self.multiply2 = nn.Multiply()
        self.add2 = nn.Add
