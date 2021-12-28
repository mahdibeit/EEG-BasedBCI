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
        '''LSTM initilization'''
        self.input_dim = 16*128
        self.hidden_dim = 16
        self.n_layers = 1
        self.batch_size = 1
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.n_layers, batch_first=True)
        self.hidden_state = torch.randn(self.n_layers, self.batch_size, self.hidden_dim)
        self.cell_state = torch.randn(self.n_layers, self.batch_size, self.hidden_dim)
        self.hidden = (self.hidden_state, self.cell_state)
        '''CNN initilization'''
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = (3,3), stride=(1,1))
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (2,2), stride=(1,1))
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (1,1), stride=(1,1))
        '''FC initilization'''
        self.fc1 = nn.Linear(16, 8)
        self.fc2 = nn.Linear(8, 4)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, data):
           
        hidden = None
        for t in range(data.size(1)):
            with torch.no_grad():
                x=data[None,:,t,:,:]
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                x = F.relu(self.conv3(x))  
                x=x.reshape(1,1,-1)
                   
            x, hidden = self.lstm(x, hidden)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.reshape(1,-1)
        return x