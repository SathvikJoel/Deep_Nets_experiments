"""
* Author : Sathvik Joel K
*
* File Name : channel_attention.py
*
* Purpose : Account for channel attention using cam from CBAM along with local attention
*
* Created On : 30 - aug -2020
*
* Bugs : --
*
"""
# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from fastai import *
from fastai.vision.all import *
from fastai.metrics import error_rate, accuracy

import sys
from src.lib.attention_mods import *
from src.local_attetion import Model


"""
Arch : Base architecture for Resnet 
Source : Taken from local_attetion.py 
Changes : Added cam to the first and second convolutional blocks 
Tests : --passed
"""
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, groups=1, base_width=64):
        super(Bottleneck, self).__init__()
        self.stride = stride
        width = int(out_channels * (base_width / 64.)) * groups

        """Observe the change"""
        self.conv1 = nn.Sequential(
            cam(in_channels),
            nn.Conv2d(in_channels, width, kernel_size=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU()
        )
        """Observe the change"""
        self.conv2 = nn.Sequential(
            cam(width),
            AttentionConv(width, width, kernel_size=7, padding=3, groups=8),
            nn.BatchNorm2d(width),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(width, self.expansion * out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.expansion * out_channels),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.stride >= 2:
            out = F.avg_pool2d(out, (self.stride, self.stride))

        out += self.shortcut(x)
        out = F.relu(out)

        return out

def ResNet50_chanspa(num_classes = 10 , stem = False):
    return Model(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, stem=stem)
