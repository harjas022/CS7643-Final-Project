import torch
import torch.nn as nn

# Ref: VGG-16 Architecture https://neurohive.io/en/popular-networks/vgg16/
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        self.max_pool = nn.MaxPool2d(2, 2)

        self.conv_2d_1_1 = nn.Conv2d(3, 64, 3, 1)
        self.conv_2d_1_2 = nn.Conv2d(64, 64, 3, 1)

        self.conv_2d_2_1 = nn.Conv2d(64, 128, 3, 1)
        self.conv_2d_2_2 = nn.Conv2d(128, 128, 3, 1)

        self.conv_2d_3_1 = nn.Conv2d(128, 256, 3, 1)
        self.conv_2d_3_2 = nn.Conv2d(256, 256, 3, 1)
        self.conv_2d_3_3 = nn.Conv2d(256, 256, 3, 1)

        self.conv_2d_4_1 = nn.Conv2d(256, 512, 3, 1)
        self.conv_2d_4_2 = nn.Conv2d(512, 512, 3, 1)
        self.conv_2d_4_3 = nn.Conv2d(512, 512, 3, 1)

        self.conv_2d_5_1 = nn.Conv2d(512, 512, 3, 1)
        self.conv_2d_5_2 = nn.Conv2d(512, 512, 3, 1)
        self.conv_2d_5_3 = nn.Conv2d(512, 512, 3, 1)

        self.linear_1 = nn.Linear(25088, 4096)
        self.linear_2 = nn.Linear(4096, 4096)
        
        self.relu = nn.ReLU()

    def forward(self, x):

        batch_size, h, w, channel = x.shape
        x = x.reshape(batch_size, channel, h, w)

        x = self.conv_2d_1_1(x)
        x = self.conv_2d_1_2(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.conv_2d_2_1(x)
        x = self.conv_2d_2_2(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.conv_2d_3_1(x)
        x = self.conv_2d_3_2(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.conv_2d_4_1(x)
        x = self.conv_2d_4_2(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.conv_2d_5_1(x)
        x = self.conv_2d_5_2(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = torch.flatten(x, start_dim=1)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)

        return x