import torch
import torch.nn as nn
from .utils.util_classes import SplitTensor, AddInQuadrature, DepthSum, ConvWH, ConvDW, ConvDH

class MixtureOfExpertsSegmentationEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.split = SplitTensor()
        self.add = AddInQuadrature()
        self.sum = DepthSum()

        self.conv_dh_1 = ConvDH(in_channels=1, out_channels=96, kernel_size=(3, 11), stride=(1, 2), padding=(1, 5))
        self.conv_dw_1 = ConvDW(in_channels=1, out_channels=96, kernel_size=(3, 11), stride=(1, 2), padding=(1, 5))
        self.conv_wh_1 = ConvWH(in_channels=1, out_channels=96, kernel_size=(11, 11), stride=(2, 2), padding=(5, 5))

        self.conv_dh_2 = ConvDH(in_channels=96, out_channels=256, kernel_size=(3, 11), stride=(1, 2), padding=(1, 5))
        self.conv_dw_2 = ConvDW(in_channels=96, out_channels=256, kernel_size=(3, 11), stride=(1, 2), padding=(1, 5))
        self.conv_wh_2 = ConvWH(in_channels=96, out_channels=256, kernel_size=(11, 11), stride=(2, 2), padding=(5, 5))

        self.conv_dh_3 = ConvDH(in_channels=256, out_channels=360, kernel_size=(3, 7), stride=(1, 1), padding=(1, 3))
        self.conv_dw_3 = ConvDW(in_channels=256, out_channels=360, kernel_size=(3, 7), stride=(1, 1), padding=(1, 3))
        self.conv_wh_3 = ConvWH(in_channels=256, out_channels=360, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))

        self.conv_dh_4 = ConvDH(in_channels=360, out_channels=512, kernel_size=(3, 7), stride=(1, 1), padding=(2, 1))
        self.conv_dw_4 = ConvDW(in_channels=360, out_channels=512, kernel_size=(3, 7), stride=(1, 1), padding=(2, 1))
        self.conv_wh_4 = ConvWH(in_channels=360, out_channels=512, kernel_size=(7, 7), stride=(1, 1), padding=(1, 1))

        self.conv_dh_5 = ConvDH(in_channels=512, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 0))
        self.conv_dw_5 = ConvDW(in_channels=512, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 0))
        self.conv_wh_5 = ConvWH(in_channels=512, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))

        self.conv_dh_6 = ConvDH(in_channels=256, out_channels=96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 0))
        self.conv_dw_6 = ConvDW(in_channels=256, out_channels=96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 0))
        self.conv_wh_6 = ConvWH(in_channels=256, out_channels=96, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))

        self.conv_dh_7 = ConvDH(in_channels=96, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=(1, 0))
        self.conv_dw_7 = ConvDW(in_channels=96, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=(1, 0))
        self.conv_wh_7 = ConvWH(in_channels=96, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        self.pool = nn.MaxPool2d((2, 2), stride=(1, 1))
        self.bn_360 = nn.BatchNorm2d(360)
        self.bn_512 = nn.BatchNorm2d(512)
        self.bn_256 = nn.BatchNorm2d(256)
        self.bn_96 = nn.BatchNorm2d(96)
        self.relu = nn.ReLU()
    
    def forward(self, X):
        outdh, outdw, outwh = self.split(X)

        outdh = self.relu(self.conv_dh_1(outdh))
        outdw = self.relu(self.conv_dw_1(outdw))
        outwh = self.relu(self.conv_wh_1(outwh))

        outdh = self.bn_256(self.relu(self.conv_dh_2(outdh)))
        outdw = self.bn_256(self.relu(self.conv_dw_2(outdw)))
        outwh = self.bn_256(self.relu(self.conv_wh_2(outwh)))

        outdh = self.pool(outdh)
        outdw = self.pool(outdw)
        outwh = self.pool(outwh)

        outdh = self.relu(self.conv_dh_3(outdh))
        outdw = self.relu(self.conv_dw_3(outdw))
        outwh = self.relu(self.conv_wh_3(outwh))

        outdh = self.bn_512(self.relu(self.conv_dh_4(outdh)))
        outdw = self.bn_512(self.relu(self.conv_dw_4(outdw)))
        outwh = self.bn_512(self.relu(self.conv_wh_4(outwh)))

        outdh = self.pool(outdh)
        outdw = self.pool(outdw)
        outwh = self.pool(outwh)

        outdh = self.relu(self.conv_dh_5(outdh))
        outdw = self.relu(self.conv_dw_5(outdw))
        outwh = self.relu(self.conv_wh_5(outwh))

        outdh = self.bn_96(self.relu(self.conv_dh_6(outdh)))
        outdw = self.bn_96(self.relu(self.conv_dw_6(outdw)))
        outwh = self.bn_96(self.relu(self.conv_wh_6(outwh)))

        # outdh = self.relu(self.conv_dh_7(outdh))
        # outdw = self.relu(self.conv_dw_7(outdw))
        # outwh = self.relu(self.conv_wh_7(outwh))

        outdh = self.pool(outdh)
        outdw = self.pool(outdw)
        outwh = self.pool(outwh)

        out = self.add(outdh, outdw, outwh)
        out = self.sum(out)
        return (out - out.min()) / (out.max() - out.min())