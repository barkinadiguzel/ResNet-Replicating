import torch.nn as nn

class FCLayer(nn.Module):
    def __init__(self, in_features, num_classes=1000):
        super(FCLayer, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)
