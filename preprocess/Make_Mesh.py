# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 13:16:30 2021

@author: mahdi
"""
import numpy as np
from utils import Global_Variable 

def Make_Mesh(Trials):
    
    '''Transforms the raw EEG signals into 2D mesh array accodding to
    the location of the electrodes in EEG headset'''
      
    # Size of the mesh
    Height=7
    Length=7
    # Location of the elctrodes 
    Electrode_Pos=Global_Variable.Electrode.Get_Elec_Postions()
    # Make the Mesh (Time, Height, Length, Trial)
    Brain_Map= np.zeros((len(Trials[0][0]),Height,Length,len(Trials)))
    #Load Data in Mesh 
    for Trial in range(len(Trials)):
        for elec in range(1,23):
            pos=Electrode_Pos.index(elec)
            Brain_Map[:,int(np.floor(pos/Length)),int(pos%Length),Trial]=(Trials[Trial,elec-1,:])
    return Brain_Map
   
if __name__=="__main__":
    pass
    
    