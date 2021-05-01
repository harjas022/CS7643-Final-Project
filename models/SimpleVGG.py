import torch
import torch.nn as nn
class SimpleVGG(nn.Module):
    def __init__(self):
        super(SimpleVGG, self).__init__()
        self.dropout = nn.Dropout(0.05)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.conv_2d_1 = nn.Conv2d(3, 64, 3, 1)
        self.conv_2d_2 = nn.Conv2d(64, 128, 3, 1)
        self.conv_2d_3 = nn.Conv2d(128, 256, 3, 1)
        self.linear_1 = nn.Linear(921600, 32)
        self.linear_2 = nn.Linear(32, 4)
        self.relu = nn.ReLU()
        
        self.gradients = None
    def forward(self, x):
        batch_size, h, w, channel = x.shape
        x = x.reshape(batch_size, channel, h, w)
        x = self.conv_2d_1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv_2d_2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv_2d_3(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        
        x.register_hook(self.activations_hook)
        return x
    
    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        print("X Size: ", x.size())
        return self.conv_2d_3(x)