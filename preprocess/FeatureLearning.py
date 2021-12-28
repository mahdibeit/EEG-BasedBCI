# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 22:21:01 2021

@author: mahdi
"""

import torch
import torch.optim as optim
from train.AEDataset import AEDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from models.Autoencoder import AE
import numpy as np



def FeatureLearning(Brain_Map):
    
    '''Load the Model'''
    model = AE(input_shape=49)
    if torch.cuda.is_available():
      model = model.to('cuda')
    model.load_state_dict(torch.load('AE_model.pt'))
    model.eval()
 
    '''Pass the data'''
    for i in range(Brain_Map.shape[0]):
      for k in range(Brain_Map.shape[3]):

        inputs = torch.from_numpy(Brain_Map[i,:,:,k])
        inputs = inputs[None,:,:,None]
        # Transfer Data to GPU if available
        if torch.cuda.is_available():
          inputs=inputs.to('cuda')
    
        # Forward Pass
        inputs=torch.tensor(inputs.float()) 
        Outputs = model(inputs)
        Brain_Map[i,:,:,k]=Outputs.cpu().data.numpy()

    return Brain_Map