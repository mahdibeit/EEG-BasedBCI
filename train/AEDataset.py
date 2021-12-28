# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 23:14:38 2021

@author: mahdi
"""

from torch.utils.data import Dataset
import torch
import numpy as np

class AEDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, Brain_Map, window=1,starting=750,Stride=1,Ending=1500):
        """
        Cropping and Windowing each trial
        """
        self.starting=starting
        self.Window = window
        self.End=Ending
        self.Stride=Stride
        self.Brain_Map = Brain_Map
        self.Trial_Length=int(np.floor((self.End-self.starting)/self.Stride))
        
    def __len__(self):
        return (self.Brain_Map.shape[3])*self.Trial_Length

    def __getitem__(self, idx):
        # Classes Between 0 and 3
        "Croping and Windowing each trial"
        Counter=int(np.floor(idx/self.Trial_Length))
        sliding=self.starting+self.Stride*int(np.floor(idx%self.Trial_Length))
        Mesh=self.Brain_Map[sliding:sliding+self.Window,:,:,Counter]
        Mesh_Tens=torch.tensor(Mesh)

        return Mesh_Tens