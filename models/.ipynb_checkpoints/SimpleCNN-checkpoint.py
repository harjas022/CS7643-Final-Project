import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    
    ## Initialization of the model
    def __init__(self):
        super(CNN, self).__init__()
        
        ## Conv2d(in_channels, out_channels, kernel_size, stride)
        self.conv1= nn.Conv2d(3, 32, 10, stride=2) 
        self.conv2= nn.Conv2d(32, 64, 3, stride=2)
        self.pool= nn.MaxPool2d(3, 3) 
        self.relu= nn.ReLU()
        self.linear1= nn.Linear(10816,1000)
        self.linear2= nn.Linear(1000, 4)
        
    
    ## Defining the forward function
    def forward(self, x):
                
        batch_size= x.shape[0]
        channel= x.shape[3]
        h= x.shape[1]
        w= x.shape[2]
        x= x.reshape(batch_size, channel, h, w)
        
        output= self.conv1(x)
        output= self.pool(output)
        output= self.relu(output)
        output= self.conv2(output)
        output= self.pool(output)
        output= self.relu(output)
                        
        output= output.reshape(batch_size, 10816)
        output= self.linear1(output)
        output= self.linear2(output)
        
        return output