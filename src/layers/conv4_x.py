import torch
import torch.nn as nn
from layers.bottleneck_block import BottleneckBlock

class Conv4_X(nn.Module):
    def __init__(self, in_channels=512, out_channels=1024):
        super(Conv4_X, self).__init__()
        self.layer1 = BottleneckBlock(in_channels, 256, out_channels, stride=2)
        self.layer2 = BottleneckBlock(out_channels, 256, out_channels, stride=1)
        self.layer3 = BottleneckBlock(out_channels, 256, out_channels, stride=1)
        self.layer4 = BottleneckBlock(out_channels, 256, out_channels, stride=1)
        self.layer5 = BottleneckBlock(out_channels, 256, out_channels, stride=1)
        self.layer6 = BottleneckBlock(out_channels, 256, out_channels, stride=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x
