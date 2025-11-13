import torch
import torch.nn as nn

class Conv1(nn.Module):
    def __init__(self):
        super(Conv1, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,  # padding = (kernel-1)//2
            bias=False
        )
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
