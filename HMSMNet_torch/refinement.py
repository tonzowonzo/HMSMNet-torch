import torch
import torch.nn as nn


def conv_bn_act(in_channels, out_channels, kernel_size, stride, padding, dilation_rate):
    conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                     stride=stride, padding=padding, dilation=dilation_rate, bias=False)
    bn = nn.BatchNorm2d(out_channels)
    act = nn.LeakyReLU()

    return nn.Sequential(conv, bn, act)


class Refinement(nn.Module):
    def __init__(self, filters):
        super(Refinement, self).__init__()
        self.conv1 = conv_bn_act(filters=filters, in_channels=4, out_channels=filters, kernel_size=3, stride=1,
                                 padding=1, dilation_rate=1)
        self.conv2 = conv_bn_act(filters=filters, in_channels=filters, out_channels=filters, kernel_size=3, stride=1,
                                 padding=1, dilation_rate=1)
        self.conv3 = conv_bn_act(filters=filters, in_channels=filters, out_channels=filters, kernel_size=3, stride=1,
                                 padding=2, dilation_rate=2)
        self.conv4 = conv_bn_act(filters=filters, in_channels=filters, out_channels=filters, kernel_size=3, stride=1,
                                 padding=3, dilation_rate=3)
        self.conv5 = conv_bn_act(filters=filters, in_channels=filters, out_channels=filters, kernel_size=3, stride=1,
                                 padding=1, dilation_rate=1)
        self.conv6 = nn.Conv2d(in_channels=filters, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, inputs):
        # inputs: [disparity, rgb, gx, gy]
        assert len(inputs) == 4

        scale_factor = inputs[1].shape[2] / inputs[0].shape[2]
        disp = nn.functional.interpolate(inputs[0], size=[inputs[1].shape[2], inputs[1].shape[3]], mode='bilinear')
        disp = disp * scale_factor

        concat = torch.cat([disp, inputs[1], inputs[2], inputs[3]], dim=1)
        delta = self.conv1(concat)
        delta = self.conv2(delta)
        delta = self.conv3(delta)
        delta = self.conv4(delta)
        delta = self.conv5(delta)
        delta = self.conv6(delta)
        disp_final = disp + delta

        return disp_final