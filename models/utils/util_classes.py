import torch
import torch.nn as nn

class SplitTensor(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, X):
        if X.ndim == 3:
            return X
        dh, dw, wh = X.sum(dim=-1), X.sum(dim=-2), X.sum(dim=-3)
        return dh.unsqueeze(1), dw.unsqueeze(1), wh.unsqueeze(1)
    
class AddInQuadrature(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, dh, dw, wh):
        dh, dw, wh = dh.squeeze(1).unsqueeze(-1), dw.squeeze(1).unsqueeze(-2), wh.squeeze(1).unsqueeze(-3)
        return torch.sqrt(dh**2 + dw**2 + wh**2)

class DepthSum(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, X):
        return torch.sqrt((X**2).sum(dim=2))

class ConvDH(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    
    def forward(self, X):
        return self.conv(X)

class ConvDW(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, X):
        return self.conv(X)

class ConvWH(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, X):
        return self.conv(X)