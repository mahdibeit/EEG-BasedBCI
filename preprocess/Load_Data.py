# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 13:14:13 2021

@author: mahdi
"""


import numpy as np
import scipy.io as sio


def Get_Data(subject,training,PATH):
    
	'''	Loads the dataset 2a of the BCI Competition IV'''

	NO_channels = 22
	NO_tests = 6*48 	
	Window_Length = 7*250 

	class_return = np.zeros(NO_tests)
	data_return = np.zeros((NO_tests,NO_channels,Window_Length))
    
	NO_valid_trial = 0
	if training:
		a = sio.loadmat(PATH+'A0'+str(subject)+'T.mat') #imports the training set
	else:
		a = sio.loadmat(PATH+'A0'+str(subject)+'E.mat') #imports the test set
	a_data = a['data']
	for ii in range(0,a_data.size):
		a_data1 = a_data[0,ii]
		a_data2=[a_data1[0,0]]
		a_data3=a_data2[0]    #Unpacks the data in the .mat
		a_X 		= a_data3[0]
		a_trial 	= a_data3[1]
		a_y 		= a_data3[2]
		a_artifacts = a_data3[5]

		for trial in range(0,a_trial.size): 
			if(a_artifacts[trial]==0):   #Loads the rials with no artifacts
				data_return[NO_valid_trial,:,:] = np.transpose(a_X[int(a_trial[trial]):(int(a_trial[trial])+Window_Length),:22])
				class_return[NO_valid_trial] = int(a_y[trial])
				NO_valid_trial +=1


	return data_return[0:NO_valid_trial,:,:], class_return[0:NO_valid_trial]


if __name__=="__main__":
    pass

