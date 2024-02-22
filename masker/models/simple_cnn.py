import torch
from torch import nn

class SimpleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(SimpleConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class SimpleMaskRCNN(nn.Module):
    def __init__(self):
        super(SimpleMaskRCNN, self).__init__()
        self.feature_extractor = nn.Sequential(
            SimpleConv(1, 16, 5),
            SimpleConv(16, 32, 3),
            SimpleConv(32, 64, 3),
        )
        self.mask_predictor = SimpleConv(64, 1)

    def forward(self, x):
        features = self.feature_extractor(x)
        masks = self.mask_predictor(features)
        return masks
