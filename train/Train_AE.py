
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 23:11:01 2021

@author: mahdi
"""

import torch
import torch.optim as optim
from train.AEDataset import AEDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from models.Autoencoder import AE
import numpy as np

def AE_Train(Brain_Map,Brain_Map_Test):
    "Set the model"
    criterion = nn.MSELoss()
    net=AE(input_shape=49)
    net = net.double()
    optimizer = optim.Adam(net.parameters(), lr=0.00001)
    Batch_Size_Train=1
    Max_Epoch=50
    #if the GPU is available load the model to GPU
    if torch.cuda.is_available():
      net = net.to('cuda')

    
    "Load data"
    dataset=AEDataset(Brain_Map)
    dataset_Test= AEDataset(Brain_Map_Test)
    trainloader = DataLoader(dataset, batch_size=Batch_Size_Train, shuffle=True)
    Valid_Loader = DataLoader(dataset_Test, batch_size=1, shuffle=True)
    
    "Training and Validation"
    min_valid_loss = np.inf
    max_acc=-np.inf
    for epoch in range(Max_Epoch):  # loop over the dataset multiple times
    
        training_loss = 0.0
      
        net.train()
        for i, data in enumerate(trainloader, 0):

            # get the inputs
            inputs = data

            # Transfer Data to GPU if available
            if torch.cuda.is_available():
              inputs = inputs.to('cuda')
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize 
            outputs = net(inputs[:,:,:,:])
            
            loss = criterion(outputs, inputs)
            # print(loss.item())
            loss.backward()
            optimizer.step()
    
            # print statistics
            training_loss += loss.item()
            if i % 100 == 90:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, training_loss / 100))
                training_loss = 0.0
        print(f' AE Training Accuracy \t {training_loss}')
        
        "Validation"
        valid_loss = 0.0
        valid_acc=0
        net.eval()     # Optional when not using Model Specific layer
        for inputs in Valid_Loader:
            # Transfer Data to GPU if available
            if torch.cuda.is_available():
              inputs = inputs.to('cuda')
              
            # Forward Pass
            inputs=torch.tensor(inputs[:,:,:,:]) 
            target = net(inputs)

            #print(pred, labels.tolist())

            # Find the Loss
            loss = criterion(target,inputs)
            # Calculate Loss
            valid_loss += loss.item()
 
        print(f'Epoch {epoch+1} \t\t AE Validation Loss: {valid_loss } ')
        
        if (valid_acc/len(Valid_Loader)) > max_acc:
            max_acc=valid_acc/len(Valid_Loader)
         
        if min_valid_loss > valid_loss:
            print(f' AE Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            torch.save(net.state_dict(), 'AE_model.pt')
        
    print('Finished Training')
    print(max_acc)