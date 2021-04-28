import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class YOLO(nn.Module):
    
    def init(self):
        super(YOLO, self).__init__()
        
        ## 24 Convolution Layers
        self.conv1 = nn.Conv2d(3, 224, 3, 3)
        self.pooling = nn.Pool2d(2, 2)
        
        
    def forward(self, x):
        
        output = self.conv1(x)
        output = self.pooling(output)
        
        return output