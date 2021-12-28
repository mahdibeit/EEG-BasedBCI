# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 09:03:41 2021

@author: mahdi
"""

import torch.nn as nn
import torch.nn.functional as F
import torch


class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        '''Encoder and decoder part of the autoencoder'''
        self.encoder1 = nn.Linear(in_features=kwargs["input_shape"], out_features=32)
        self.encoder2 = nn.Linear(in_features=32, out_features=8)
        self.decoder1 = nn.Linear(in_features=8, out_features=32)
        self.decoder2 = nn.Linear(in_features=32, out_features=kwargs["input_shape"])
        
    def forward(self, x):

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.encoder1(x))
        x = F.relu(self.encoder2(x))
        x = F.relu(self.decoder1(x))
        x = (self.decoder2(x)) 
        x=x.reshape(-1,1,7,7) # reshape back to the original shape

        return x
        
    
