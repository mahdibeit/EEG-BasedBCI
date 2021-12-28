# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 23:11:01 2021

@author: mahdi
"""

import torch
import torch.optim as optim
from train.MeshDataset import MeshDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from models.CNN3D import Net
import numpy as np

def Train(Brain_Map,Labels,Brain_Map_Test,Labels_Test ):
    "Set the model"
    criterion = nn.CrossEntropyLoss()
    net=Net()
    net = net.double()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    Batch_Size_Train=10
    Max_Epoch=1000
    #if the GPU is available load the model to GPU
    if torch.cuda.is_available():
              net = net.to('cuda')

    
    "Load data"
    dataset=MeshDataset(Brain_Map, Labels)
    dataset_Test=MeshDataset(Brain_Map_Test, Labels_Test)
    trainloader = DataLoader(dataset, batch_size=Batch_Size_Train, shuffle=True)
    Valid_Loader = DataLoader(dataset_Test, batch_size=1, shuffle=True)
    
    "Training and Validation"
    min_valid_loss = np.inf
    max_acc=-np.inf
    for epoch in range(Max_Epoch):  # loop over the dataset multiple times
    
        training_loss = 0.0
        pred_acc=0
        net.train()
        for i, data in enumerate(trainloader, 0):

          

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

           

            # Transfer Data to GPU if available
            if torch.cuda.is_available():
              inputs = inputs.to('cuda')
              labels = labels.to('cuda')

    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize 
            outputs = net(inputs[:,None,:,:,:])
            _, pred=torch.max(outputs, 1)
            
            for k in range(len(pred.tolist())):
                 if labels.tolist()[k] == pred.tolist()[k]: pred_acc+=1
            loss = criterion(outputs, labels.long())
            # print(loss.item())
            loss.backward()
            optimizer.step()
    
            # print statistics
            training_loss += loss.item()
            if i % 100 == 90:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, training_loss / 100))
                training_loss = 0.0
        print(f'Training Accuracy \t {pred_acc/(len(trainloader)*Batch_Size_Train)}')
        
        "Validation"
        valid_loss = 0.0
        valid_acc=0
        net.eval()     # Optional when not using Model Specific layer
        for inputs, labels in Valid_Loader:
            # Transfer Data to GPU if available
            if torch.cuda.is_available():
              inputs = inputs.to('cuda')
              labels = labels.to('cuda')
             
            # Forward Pass
            inputs=torch.tensor(inputs[:,None,:,:,:]) 
            target = net(inputs)
            _, pred=torch.max(target, 1)
            #print(pred, labels.tolist())
            for k in range(len(pred.tolist())):
                 if labels.tolist()[k] == pred.tolist()[k]: valid_acc+=1
            # Find the Loss
            loss = criterion(target,labels.long())
            # Calculate Loss
            valid_loss += loss.item()
 
        print(f'Epoch {epoch+1} \t\t Validation Loss: {valid_loss / len(Valid_Loader)} \n Valid Accuracy \t {valid_acc/len(Valid_Loader)}')
        
        if (valid_acc/len(Valid_Loader)) > max_acc:
            max_acc=valid_acc/len(Valid_Loader)
         
        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
        
    print('Finished Training')
    print(max_acc)