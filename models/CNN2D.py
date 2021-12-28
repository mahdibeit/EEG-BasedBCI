# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 22:57:20 2021

@author: mahdi
"""

import torch.nn as nn
import torch.nn.functional as F
import torch


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = (3,3), stride=(1,1),padding='same')
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3,3), stride=(1,1),padding='same')
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3), stride=(1,1),padding='same')
        self.fc1 = nn.Linear(64 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.dropout(self.fc2(x)))
        x = self.fc3(x)
        return x