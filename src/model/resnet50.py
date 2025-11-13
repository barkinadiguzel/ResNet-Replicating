import torch
import torch.nn as nn
from layers.conv1 import Conv1
from layers.conv2_x import Conv2_X
from layers.conv3_x import Conv3_X
from layers.conv4_x import Conv4_X
from layers.conv5_x import Conv5_X
from layers.pool_layers.avgpool_layer import AvgPoolLayer
from layers.flatten_layer import FlattenLayer
from layers.fc_layer import FCLayer

class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet50, self).__init__()
        self.conv1 = Conv1()
        self.conv2_x = Conv2_X()
        self.conv3_x = Conv3_X()
        self.conv4_x = Conv4_X()
        self.conv5_x = Conv5_X()
        self.avgpool = AvgPoolLayer()
        self.flatten = FlattenLayer()
        self.fc = FCLayer(num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
