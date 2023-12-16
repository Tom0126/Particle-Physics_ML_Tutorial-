#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/3 10:24
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : resnet.py
# @Software: PyCharm


from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torchsummary import summary
from Data.loader import data_loader
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, short_cut=True):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.dp1=nn.Dropout2d(p=0.2)

        self.short_cut=short_cut

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out=self.dp1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.short_cut:
            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity


        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, short_cut=True ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dp1 = nn.Dropout2d(p=0.2)
        self.dp2 = nn.Dropout2d(p=0.2)

        self.short_cut=short_cut

    def forward(self, x):



        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.dp1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dp2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.short_cut:
            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers,  start_planes=40,num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, short_cut=True, first_kernal=7,first_stride=2, first_padding=3):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.start_planes=start_planes
        self.inplanes = 64
        self.dilation = 1
        self.short_cut=short_cut

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.avgpool0=nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv1 = nn.Conv2d(self.start_planes, self.inplanes, kernel_size=first_kernal, stride=first_stride, padding=first_padding,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer,short_cut=self.short_cut))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes=self.inplanes,planes=planes,groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, short_cut=self.short_cut))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 1)
                m.bias.data.zero_()



class ResNet_Avg(ResNet):

    def __init__(self, block, layers,  start_planes=40,num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, short_cut=True, first_kernal=7, first_stride=2, first_padding=3):
        super().__init__(block, layers,  start_planes ,num_classes, zero_init_residual,
                 groups, width_per_group, replace_stride_with_dilation,
                 norm_layer, short_cut, first_kernal, first_stride, first_padding)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x=self.avgpool0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

if __name__ == '__main__':
    paras={'block':Bottleneck,
           'layers': [2,2,2,2],
           'num_classes':4,
           'start_planes':40,
           'short_cut': False,
           'first_kernal': 5,
           }

    resnet18=ResNet_Avg(**paras)
    resnet18.initialize_weights()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet18 = resnet18.to(device)
    summary(resnet18, (40, 6, 6))

    # t=torch.Tensor(np.ones((2,40,18,18)))
    # t=t.to(device)
    # t=resnet18(t)
    # print(t)
    #
    # img_path = '/hpcfs/cepc/higgsgpu/siyuansong/PID/data/beam_2023/beam_xia/Test/imgs.npy'
    # label_path = '/hpcfs/cepc/higgsgpu/siyuansong/PID/data/beam_2023/beam_xia/Test/labels.npy'
    # loader = data_loader(img_path, label_path, num_workers=0, mean_std_static=True)
    # for i, (img, label) in enumerate(loader):
    #     print('img:{} label:{}'.format(img.shape, label.shape))
    #     t=resnet18(img)
    #     print(t)
    #     if i == 0:
    #         break
    #
    # pass
