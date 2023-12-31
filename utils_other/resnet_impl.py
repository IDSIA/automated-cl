# File copied from https://github.com/yinboc/few-shot-meta-baseline/blob/master/models/resnet12.py
# Used with minor modifications.
# =============================================================================
# MIT License
#
# Copyright (c) 2020 Yinbo Chen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch.nn as nn


def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 3, padding=1, bias=False)


def conv1x1(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 1, bias=False)


def norm_layer(planes, instance_norm=False):
    if instance_norm:
        return nn.InstanceNorm2d(planes)
    else:
        return nn.BatchNorm2d(planes)


class Block(nn.Module):

    def __init__(self, inplanes, planes, downsample, dropout=0.0, instance_norm=False):
        super().__init__()

        self.relu = nn.LeakyReLU(0.1)

        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes, instance_norm)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, instance_norm)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = norm_layer(planes, instance_norm)
        self.dropout = nn.Dropout(dropout)

        self.downsample = downsample

        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        out = self.maxpool(out)

        out = self.dropout(out)

        return out


class DropInBlock(nn.Module):

    def __init__(self, inplanes, planes, downsample, dropout=0.0, instance_norm=False):
        super().__init__()

        self.relu = nn.LeakyReLU(0.1)

        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes, instance_norm)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, instance_norm)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = norm_layer(planes, instance_norm)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.downsample = downsample

        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.dropout2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        out = self.maxpool(out)

        out = self.dropout3(out)

        return out


class Drop2dBlock(nn.Module):

    def __init__(self, inplanes, planes, downsample, dropout=0.0, instance_norm=False):
        super().__init__()

        self.relu = nn.LeakyReLU(0.1)

        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes, instance_norm)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, instance_norm)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = norm_layer(planes, instance_norm)
        self.dropout = nn.Dropout2d(dropout)

        self.downsample = downsample

        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        out = self.maxpool(out)

        out = self.dropout(out)

        return out


class Drop2dInBlock(nn.Module):

    def __init__(self, inplanes, planes, downsample, dropout=0.0, instance_norm=False):
        super().__init__()

        self.relu = nn.LeakyReLU(0.1)

        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes, instance_norm)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, instance_norm)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = norm_layer(planes, instance_norm)

        self.dropout1 = nn.Dropout2d(dropout)
        self.dropout2 = nn.Dropout2d(dropout)
        self.dropout3 = nn.Dropout2d(dropout)

        self.downsample = downsample

        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.dropout2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        out = self.maxpool(out)

        out = self.dropout3(out)

        return out


class ResNet12(nn.Module):

    def __init__(self, channels, dropout, dropout_type='base', instance_norm=False):
        super().__init__()

        self.inplanes = 3

        self.layer1 = self._make_layer(channels[0], dropout, dropout_type, instance_norm)
        self.layer2 = self._make_layer(channels[1], dropout, dropout_type, instance_norm)
        self.layer3 = self._make_layer(channels[2], dropout, dropout_type, instance_norm)
        self.layer4 = self._make_layer(channels[3], dropout, dropout_type, instance_norm)

        self.out_dim = channels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, dropout, dropout_type='base', instance_norm=False):
        downsample = nn.Sequential(
            conv1x1(self.inplanes, planes),
            norm_layer(planes, instance_norm),
        )
        if dropout_type == 'base':
            block = Block(self.inplanes, planes, downsample, dropout, instance_norm)
        elif dropout_type == 'inblock':
            block = DropInBlock(self.inplanes, planes, downsample, dropout, instance_norm)
        elif dropout_type == '2d':
            block = Drop2dBlock(self.inplanes, planes, downsample, dropout, instance_norm)
        elif dropout_type == '2d_inblock':
            block = Drop2dInBlock(self.inplanes, planes, downsample, dropout, instance_norm)
        else:
            assert False, f"Unknown dropout_type: {dropout_type}"
        self.inplanes = planes
        return block

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.view(x.shape[0], x.shape[1], -1).mean(dim=2)

        return x


def resnet12(dropout=0.0):  # output feat dim 512
    return ResNet12([64, 128, 256, 512], dropout)


def resnet12_base(dropout=0.0, use_big=False, dropout_type='base', instance_norm=False):
    if use_big:
        return ResNet12([64, 128, 256, 512], dropout, dropout_type, instance_norm)
    else:  # output feat dim 256
        return ResNet12([64, 96, 128, 256], dropout, dropout_type, instance_norm)
