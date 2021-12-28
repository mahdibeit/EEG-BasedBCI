# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 23:16:18 2021

@author: mahdi
"""

import numpy as np
from scipy.signal import butter, lfilter,filtfilt,iirnotch

def Normalize(Subject):

    '''Filter the data '''
    Subject=Subject[:] #make a copy of the input
    def butter_bandpass(lowcut, highcut, fs, order=10):
      nyq = 0.5 * fs
      low = lowcut / nyq
      high = highcut / nyq
      b, a = butter(order, [low, high], btype='band')
      return b, a

    def filter_data(data, lowcut, highcut, fs, order=10, notch=50.0,filter_quality=30.0):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order) # Band pass filter
        y = lfilter(b, a, data)
        b, a = iirnotch(notch, filter_quality, fs) #Notch filter at 50Hz
        y = filtfilt(b, a, data)
        return y

    Subject=filter_data(Subject,0.5,100,250)
    
    """Normalize the data for each channel for each subject and all the trials"""
    
    "Stack the trials"
    stack_subject=np.concatenate(Subject[:,:,750:1500], axis=1)
    
    "Make the data to array andd calculate SUM and STD"
    STD= np.std(stack_subject,1)
    AVE=np.average(stack_subject,1)
       
    "Calcualte the zero score after the cue in 750"
    for trial in range(len(Subject)):
        for channel in range(len(Subject[0])):
            for sample in range(len(Subject[0][0])):
                Subject[trial,channel,sample]=Subject[trial,channel,sample]/STD[channel]-AVE[channel]
    return Subject

if __name__=='__main__':
    b=np.array([[[1,3,3,4,5],[6,7,8,9,10]], [[11,13,13,14,15],[16,17,18,19,20]]])
    Normalize(b)
    
    
            
    
    
    
        