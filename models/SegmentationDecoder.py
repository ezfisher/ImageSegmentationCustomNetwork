import torch
from torch import nn

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tconv1 = nn.ConvTranspose2d(in_channels=96, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.tconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.tconv3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(5, 5), stride=(2, 2), padding=(0, 0))
        self.tconv4 = nn.ConvTranspose2d(in_channels=256, out_channels=96, kernel_size=(5, 5), stride=(2, 2), padding=(0, 0))
        self.tconv5 = nn.ConvTranspose2d(in_channels=96, out_channels=1, kernel_size=(7, 7), stride=(1, 1), padding=(0, 0))
        self.tconv6 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(9, 9), stride=(1, 1), padding=(0, 0))
        self.tconv7 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(10, 10), stride=(1, 1), padding=(0, 0))

        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(512)
        self.bn4 = nn.BatchNorm2d(96)
        self.bn7 = nn.BatchNorm2d(1)

    def forward(self, X):
        out = self.relu(self.tconv1(X))
        out = self.bn2(self.relu(self.tconv2(out)))

        out = self.relu(self.tconv3(out))
        out = self.bn4(self.relu(self.tconv4(out)))

        out = self.relu(self.tconv5(out))
        out = self.relu(self.tconv6(out))
        out = self.bn7(self.relu(self.tconv7(out)))
        return (out - out.min()) / (out.max() - out.min())