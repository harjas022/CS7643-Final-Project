import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class YOLO(nn.Module):
    
    def __init__(self):
        super(YOLO, self).__init__()
        
        # First
        self.conv1 = nn.Conv2d(3, 32, 3, 1, padding=1)
        self.pooling = nn.AvgPool2d(2, 2)
        
        # Second
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        
        # Third
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 64, 1, 1)
        
        # Fourth
        self.conv5 = nn.Conv2d(128, 256, 3, 1, padding=1)
        self.conv6 = nn.Conv2d(256, 128, 1, 1)
        
        # Fifth
        self.conv7 = nn.Conv2d(256, 512, 3, 1, padding=1)
        self.conv8 = nn.Conv2d(512, 256, 1, 1)
        
        # Sixth
        self.conv9 = nn.Conv2d(512, 1024, 3, 1, padding=1)
        self.conv10 = nn.Conv2d(1024, 512, 1, 1)
        
        # Final
        self.conv11 = nn.Conv2d(1024, 1000, 1, 1)
        
        # FC Layer and Softmax
        self.FC = nn.Linear(1000, 4)
        
        
            
    def forward(self, x):
        
        batch_size, h, w, channel = x.shape
        x= x.reshape(batch_size, channel, h, w)
        
        output = self.conv1(x) # 500
        output = self.pooling(output) # 250
        
        output = self.conv2(output) # 250
        output = self.pooling(output) # 125
        
        output = self.conv3(output) # 125
        output = self.conv4(output) # 125
        output = self.conv3(output) # 125
        output = self.pooling(output) # 62
        
        output = self.conv5(output) # 62
        output = self.conv6(output) # 62
        output = self.conv5(output) # 62
        output = self.pooling(output) # 31
        
        output = self.conv7(output) # 31
        output = self.conv8(output) # 31
        output = self.conv7(output) # 31
        output = self.conv8(output) # 31
        output = self.conv7(output) # 31
        output = self.pooling(output) # 15
        
        output = self.conv9(output) # 15
        output = self.conv10(output) # 15
        output = self.conv9(output) # 15
        output = self.conv10(output) # 15
        output = self.conv9(output) # 15
        
        output = self.conv11(output)
        output = output.mean([2, 3])
        
        output = self.FC(output)
        
        return output