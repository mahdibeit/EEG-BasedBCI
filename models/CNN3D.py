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
        super(Net, self).__init__()
        self.conv1 = nn.Conv3d(in_channels = 1, out_channels = 16, kernel_size = (3,2,2), stride=(1,1,1),padding=(1,1,1))
        self.Max1=torch.nn.MaxPool3d(kernel_size=(4,2,2), stride=None)

        self.conv2 = nn.Conv3d(in_channels = 16, out_channels = 32, kernel_size = (3,2,2), stride=(1,1,1),padding=(1,1,1))
        self.Max2=torch.nn.MaxPool3d(kernel_size=(4,2,2), stride=None)

        #self.conv2 = nn.Conv3d(in_channels = 8, out_channels = 16, kernel_size = (30,3,3), stride=(1,1,1),padding=(0,1,1))
        #self.conv3 = nn.Conv3d(in_channels = 16, out_channels = 32, kernel_size = (1,2,2), stride=(1,2,2),padding=(4,2,2))
        self.fc1 = nn.Linear(128, 4)
        #self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(4, 4)
        #self.batch1=torch.nn.BatchNorm3d(1)
        #self.batch2=torch.nn.BatchNorm3d(8)
        #self.batch3=torch.nn.BatchNorm3d(16)
        #self.batch4=torch.nn.BatchNorm1d(784000)
        #self.batch5=torch.nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = (self.Max1(x))

        x = F.relu(self.conv2(x))
        x = (self.Max2(x))


        #x=self.dropout(x)
        #x = F.relu(self.conv2(x))
        #x=self.dropout(x)
        #x = F.relu(self.conv3(self.batch3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1((x)))
        #x=self.dropout(x)
        #x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x