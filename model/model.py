from model.config import N_CLASSES, START_FRAME
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from utils.common import ConvBlock, DeconvBlock, DoubleConvBlock, ResidualBlock

class UNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=N_CLASSES, start_fm=START_FRAME):
        super(UNet, self).__init__()
        # Input 1x128x128

        # Maxpool 
        self.pool = nn.MaxPool2d((2,2))

        # Transpose conv
        self.deconv_4  = nn.ConvTranspose2d(start_fm*16, start_fm*8, 2, 2)
        self.deconv_3  = nn.ConvTranspose2d(start_fm*8, start_fm*4, 2, 2)
        self.deconv_2  = nn.ConvTranspose2d(start_fm*4, start_fm*2, 2, 2)
        self.deconv_1  = nn.ConvTranspose2d(start_fm*2, start_fm, 2, 2)
        
        # Encoder 
        self.encoder_1 = DoubleConvBlock(in_channels, start_fm, kernel=3)
        self.encoder_2 = DoubleConvBlock(start_fm, start_fm*2, kernel=3)
        self.encoder_3 = DoubleConvBlock(start_fm*2, start_fm*4, kernel=3)
        self.encoder_4 = DoubleConvBlock(start_fm*4, start_fm*8, kernel=3)

        # Middle
        self.middle = DoubleConvBlock(start_fm*8, start_fm*16)
        
        # Decoder
        self.decoder_4 = DoubleConvBlock(start_fm*16, start_fm*8)
        self.decoder_3 = DoubleConvBlock(start_fm*8, start_fm*4)
        self.decoder_2 = DoubleConvBlock(start_fm*4, start_fm*2)
        self.decoder_1 = DoubleConvBlock(start_fm*2, start_fm)

        self.conv_last = nn.Conv2d(start_fm, n_classes, 1)

    def forward(self, x):
        # Encoder
        conv1 = self.encoder_1(x)
        x     = self.pool(conv1)

        conv2 = self.encoder_2(x)
        x     = self.pool(conv2)

        conv3 = self.encoder_3(x)
        x     = self.pool(conv3)

        conv4 = self.encoder_4(x)
        x     = self.pool(conv4)

        # Middle
        x     = self.middle(x)

        # Decoder
        x     = self.deconv_4(x)
        x     = torch.cat([conv4, x], dim=1)
        x     = self.decoder_4(x)

        x     = self.deconv_3(x)
        x     = torch.cat([conv3, x], dim=1)
        x     = self.decoder_3(x)

        x     = self.deconv_2(x)
        x     = torch.cat([conv2, x], dim=1)
        x     = self.decoder_2(x)

        x     = self.deconv_1(x)
        x     = torch.cat([conv1, x], dim=1)
        x     = self.decoder_1(x)
        
        out   = self.conv_last(x)
        return out

class UNet_ResNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=N_CLASSES, dropout=0.5, start_fm=START_FRAME):
        super(UNet_ResNet).__init__()
        
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.dropout = dropout

        self.encoder_1 = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(in_channels, 64, (3,3), padding_mode='same')),
                ('res1_1', ResidualBlock(64)),
                ('res1_2', ResidualBlock(64, batch_activation=True)),
                ('maxpool1', nn.MaxPool2d((2,2))),
                ('dropout1', nn.Dropout2d(self.dropout/2)),
            ])
        )

        self.encoder_2 = nn.Sequential(
            OrderedDict([
                ('conv2', nn.Conv2d(in_channels, 128, (3,3), padding_mode='same')),
                ('res2_1', ResidualBlock(128)),
                ('res2_2', ResidualBlock(128, batch_activation=True)),
                ('maxpool2', nn.MaxPool2d((2,2))),
                ('dropout2', nn.Dropout2d(self.dropout)),
            ])
        )

        self.encoder_3 = nn.Sequential(
            OrderedDict([
                ('conv3', nn.Conv2d(in_channels, 256, (3,3), padding_mode='same')),
                ('res3_1', ResidualBlock(256)),
                ('res3_2', ResidualBlock(256, batch_activation=True)),
                ('maxpool3', nn.MaxPool2d((2,2))),
                ('dropout3', nn.Dropout2d(self.dropout)),
            ])
        )
        
        self.encoder_4 = nn.Sequential(
            OrderedDict([
                ('conv4', nn.Conv2d(in_channels, 512, (3,3), padding_mode='same')),
                ('res4_1', ResidualBlock(512)),
                ('res4_2', ResidualBlock(512, batch_activation=True)),
                ('maxpool4', nn.MaxPool2d((2,2))),
                ('dropout4', nn.Dropout2d(self.dropout)),
            ])
        )

        self.middle = nn.Sequential(
            OrderedDict([
                ('conv5', nn.Conv2d(in_channels, 1024, (3,3), padding_mode='same')),
                ('res5_1', ResidualBlock(1024)),
                ('res5_2', ResidualBlock(1024, batch_activation=True)),
            ])
        )
        
        self.upconv4 = nn.Sequential(OrderedDict([('upconv4', DeconvBlock(1024))]))

        self.deconder_4 = nn.Sequential(
            OrderedDict([
                ('dropout4', nn.Dropout2d(self.dropout)),
                ('deconv4', nn.Conv2d(512, 512, (3,3))),
                ('res4_1', ResidualBlock(512)),
                ('res4_2', ResidualBlock(512, batch_activation=True)),
            ])
        )

        self.upconv3 = nn.Sequential(OrderedDict([('upconv3', DeconvBlock(512))]))

        self.deconder_3 = nn.Sequential(
            OrderedDict([
                ('dropout3', nn.Dropout2d(self.dropout)),
                ('deconv3', nn.Conv2d(256, 256, (3,3))),
                ('res3_1', ResidualBlock(256)),
                ('res3_2', ResidualBlock(256, batch_activation=True)),
            ])
        )

        self.upconv2= nn.Sequential(OrderedDict([('upconv2', DeconvBlock(256))]))

        self.deconder_2 = nn.Sequential(
            OrderedDict([
                ('dropout2', nn.Dropout2d(self.dropout)),
                ('deconv2', nn.Conv2d(128, 128, (3,3))),
                ('res2_1', ResidualBlock(128)),
                ('res2_2', ResidualBlock(128, batch_activation=True)),
            ])
        )

        self.upconv1 = nn.Sequential(OrderedDict([('upconv1', DeconvBlock(128))]))

        self.deconder_1 = nn.Sequential(
            OrderedDict([
                ('upconv1', DeconvBlock(128)),
                ('dropout1', nn.Dropout2d(self.dropout)),
                ('deconv1_1', nn.Conv2d(64, 64, (3,3))),
                ('res1_1', ResidualBlock(64)),
                ('res1_2', ResidualBlock(64, batch_activation=True)),
                ('deconv1_2', ConvBlock(64, self.n_classes, kernel=(1,1), activation=False))
            ])
        )

    def forward(self, x):
        # Encoder
        # 1-64
        encoder_1 = self.encoder_1(x)
        # 64-128
        encoder_2 = self.encoder_2(encoder_1)
        # 128-256
        encoder_3 = self.encoder_3(encoder_2)
        # 256-512
        encoder_4 = self.encoder_4(encoder_3)
        
        # 512-1024
        x = self.middle(encoder_4)

        # Decoder
        # 1024-512
        x = self.upconv4(x, encoder_4)
        x = self.decoder_4(x)
        # 512-256
        x = self.upconv3(x, encoder_3)
        x = self.decoder_3(x)
        # 256-128
        x = self.upconv2(x, encoder_2)
        x = self.decoder_2(x)
        #128-64-2
        x = self.upconv1(x, encoder_1)
        x = self.decoder_1(x)

        return x