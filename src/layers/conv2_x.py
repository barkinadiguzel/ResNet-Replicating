import torch
import torch.nn as nn
from layers.bottleneck_block import BottleneckBlock

class Conv2_X(nn.Module):
    def __init__(self, in_channels=64, out_channels=256):
        super(Conv2_X, self).__init__()
        self.layer1 = BottleneckBlock(in_channels, 64, out_channels, stride=1)
        self.layer2 = BottleneckBlock(out_channels, 64, out_channels, stride=1)
        self.layer3 = BottleneckBlock(out_channels, 64, out_channels, stride=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
