# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 21:53:42 2021

@author: mahdi
"""



""" Stores the location of electrodes in EEG headset"""
class Electrode(object):
    def Get_Elec_Postions():
        return [0,0,0,1,0,0,0, 0,2,3,4,5,6,0, 7,8,9,10,11,12,13, 0,14,15,16,17,18,0, 0,0,19,20,21,0,0, 0,0,0,22,0,0,0, 0,0,0,0,0,0,0]
        
if __name__=='__main__':
    print(Electrode.Get_Elec_Postions())