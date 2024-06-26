import torch
from torch import nn

class ControlSegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11, 11), stride=(2, 2), padding=(5, 5))
        self.conv_2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(11, 11), stride=(2, 2), padding=(4, 4))
        self.conv_3 = nn.Conv2d(in_channels=256, out_channels=360, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
        self.conv_4 = nn.Conv2d(in_channels=360, out_channels=512, kernel_size=(7, 7), stride=(1, 1), padding=(1, 1))
        self.conv_5 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.conv_6 = nn.Conv2d(in_channels=256, out_channels=96, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.conv_7 = nn.Conv2d(in_channels=96, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        self.pool = nn.MaxPool2d((2, 2), stride=(1, 1))
        self.bn_512 = nn.BatchNorm2d(512)
        self.bn_256 = nn.BatchNorm2d(256)
        self.bn_96 = nn.BatchNorm2d(96)
        self.relu = nn.ReLU()

    def forward(self, X):
        out = self.relu(self.conv_1(X))
        out = self.bn_256(self.relu(self.conv_2(out)))
        out = self.pool(out)

        out = self.relu(self.conv_3(out))
        out = self.bn_512(self.relu(self.conv_4(out)))
        out = self.pool(out)

        out = self.relu(self.conv_5(out))
        out = self.bn_96(self.relu(self.conv_6(out)))
        # out = self.conv_7(out)
        out = self.pool(out)

        return out