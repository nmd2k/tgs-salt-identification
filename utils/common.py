import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import Conv2d

class BatchActivate(nn.Module):
    def __init__(self, in_channels):
        super(BatchActivate, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        return F.relu(self.norm(x))

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, padding=1, stride=1, activation=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                            kernel_size=kernel, stride=stride, padding=padding)
        self.batchnorm  = BatchActivate(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.activation:
            x = self.batchnorm(x)
        return x

class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, padding=1, stride=1):
        super(DoubleConvBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel, padding, stride)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel, padding, stride)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, kernel=(2,2)):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels//2,
                                        kernel_size=kernel)

    def forward(self, x1, x2):
        x1 = self.deconv(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x =  torch.cat([x1, x2], dim=1)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, batch_activation=False):
        super(ResidualBlock).__init__()
        self.batch_activation = batch_activation
        self.norm = BatchActivate(in_channels)
        self.conv1 = ConvBlock(in_channels, in_channels)
        self.conv2 = ConvBlock(in_channels, in_channels, activation=False)

    def forward(self, input):
        x = self.norm1(input)
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.cat([x, input], dim=1)
        if self.batch_activation:
            x = self.norm(x)
        return x
