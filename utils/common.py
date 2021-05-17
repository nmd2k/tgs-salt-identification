import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import Conv2d

class BatchActivate(nn.Module):
    def __init__(self, in_channels):
        super(BatchActivate).__init__()
        self.norm = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x = F.relu(self.norm(x))
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=(3,3), activation=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel)
        self.batchlayer = BatchActivate(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.activation:
            self.batchlayer(x)
        return x

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, kernel=(2,2)):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels/2,
                                        kernel_size=kernel)

    def forward(self, x1, x2):
        x_temp = self.deconv(x1)
        x =  torch.cat([x_temp, x2], dim=1)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=(3,3), batch_activation=False):
        super(ResidualBlock).__init__()
        self.batch_activation = batch_activation
        self.norm_relu = BatchActivate(in_channels)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size)

    def forward(self, input):
        x = self.norm_relu(input)
        x = self.conv(x)
        x = F.relu(self.conv(x))
        x = torch.cat([x, input], dim=1)
        if self.batch_activation:
            x = self.norm(x)
        return x
