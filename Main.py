# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 15:36:20 2021

@author: mahdi
"""
from preprocess.Load_Data import Get_Data
from preprocess.Make_Mesh import Make_Mesh
from utils.Visualize import Visualize
from train.Train2D import Train 
from preprocess.Normalize import Normalize
from train.Train_AE import AE_Train
from preprocess.FeatureLearning import FeatureLearning


def main():
    '''Load the training data and test data'''
    Trials, Label= Get_Data(1,True, 'DataSet/')
    Trials_test, Label_test= Get_Data(1,False, 'DataSet/')  
    
    '''Normalize and filter the data'''
    Trials=Normalize(Trials)
    Trials_test=Normalize(Trials_test)
    
    '''2D representation of data'''
    Brain_Map=Make_Mesh(Trials)
    Brain_Map_Test=Make_Mesh(Trials_test)
    
    '''Visualize the data'''
    Visualize(Brain_Map[950,:,:,20])
    
    '''Feature learning of the 2D mesh'''
    AE_Train(Brain_Map, Brain_Map_Test)
    Brain_Map=FeatureLearning(Brain_Map)
    Brain_Map_Test=FeatureLearning(Brain_Map_Test)
    
    '''Train and test the data'''
    Train(Brain_Map[:,:,:,:240],Label[:240],Brain_Map[:,:,:,240:],Label[240:])


if __name__=='__main__':
    main()
    