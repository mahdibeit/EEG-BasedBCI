# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 23:24:35 2021

@author: mahdi
"""
from torch.utils.data import Dataset
import torch
import numpy as np

class MeshDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, Brain_Map, Labels, transform=None, window=3,starting=800,Stride=1,Ending=1400):
        """
        Cropping and Windowing each trial
        """
        self.starting=starting
        self.Window = window
        self.End=Ending
        self.Stride=Stride
        self.Labels = Labels
        self.Brain_Map = Brain_Map
        self.transform = transform
        self.Trial_Length=int(np.floor((self.End-self.starting)/self.Stride))

    def __len__(self):
        return len(self.Labels)*self.Trial_Length

    def __getitem__(self, idx):
        # Classes Between 0 and 3
        "Croping and Windowing the each trial"
        Counter=int(np.floor(idx/self.Trial_Length))
        sliding=self.starting+self.Stride*int(np.floor(idx%self.Trial_Length))
        class_id= torch.tensor(self.Labels[Counter]-1)
        Mesh=self.Brain_Map[sliding:sliding+self.Window,:,:,Counter]
        Mesh_Tens=torch.tensor(Mesh)

        return Mesh_Tens,class_id