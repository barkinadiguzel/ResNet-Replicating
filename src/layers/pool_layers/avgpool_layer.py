import torch
import torch.nn as nn

class AvgPoolLayer(nn.Module):
    def __init__(self, kernel_size=7):  # ResNet-50’de 7x7 feature map sonrası
        super(AvgPoolLayer, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x):
        return self.pool(x)
