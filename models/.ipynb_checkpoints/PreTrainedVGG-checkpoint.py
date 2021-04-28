import torch
import torch.nn as nn
import torchvision.models as models

class PreTrainedVGG(nn.Module):
    def __init__(self):
        super(PreTrainedVGG, self).__init__()

        self.vgg = models.vgg16(pretrained=True)
        
        self.linear = nn.Linear(1000, 4)
        
    def forward(self, x):
        batch_size, h, w, channel = x.shape
        x = x.reshape(batch_size, channel, h, w)
        
        x = self.vgg(x)

        x = self.linear(x)

        return x