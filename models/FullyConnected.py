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

        self.fc1 = nn.Linear(30*23, 1024*4)
        self.fc2 = nn.Linear(1024*4, 1024*4*2)
        self.fc3 = nn.Linear(1024*4*2, 1024)
        self.fc4 = nn.Linear(1024, 4)
        self.batch1=torch.nn.BatchNorm1d(30*7*7)
        self.batch2=torch.nn.BatchNorm1d(1024*4)
        self.batch3=torch.nn.BatchNorm1d(1024*4*2)
        self.batch4=torch.nn.BatchNorm1d(1024)

    def forward(self, x):
        x = torch.cat([x[:,0,3,:], x[:,0,8:13],x[14:22],x[23:29],x[30:33],x[38]])
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        #x=  self.batch1(x)
        print(x.size())
        x = torch.cat([x[3], x[8:13],x[14:22],x[23:29],x[30:33],x[38]])
        x = F.relu(self.fc1(x))
        x=  self.batch2(x)
        x = F.relu(self.fc2(x))
        x=  self.batch3(x)
        x = F.relu(self.fc3(x))
        x=  self.batch4(x)
        x = F.relu(self.fc4(x))
        


        return x