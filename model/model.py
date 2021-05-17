import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from utils.common import ConvBlock, DeconvBlock, ResidualBlock

class UNet(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(UNet, self).__init__()
        
        self.in_channels = in_channels
        self.n_classes = n_classes

        # Encoder Block 1
        self.encoder_1 = nn.Sequential(
            OrderedDict([
                ('conv1_1', ConvBlock(in_channels, 64)),
                ('conv1_2', ConvBlock(64, 64)),
                ('maxpool1', nn.MaxPool2d((2, 2)))
            ])
        )

        # Encoder Block 2
        self.encoder_2 = nn.Sequential(
            OrderedDict([
                ('conv2_1', ConvBlock(64, 128)),
                ('conv2_2', ConvBlock(128, 128)),
                ('maxpool2', nn.MaxPool2d((2, 2)))
            ])
        )

        # Encoder Block 3
        self.encoder_3 = nn.Sequential(
            OrderedDict([
                ('conv3_1', ConvBlock(128, 256)),
                ('conv3_2', ConvBlock(256, 256)),
                ('maxpool3', nn.MaxPool2d((2, 2)))
            ])
        )

        # Encoder Block 4
        self.encoder_4 = nn.Sequential(
            OrderedDict([
                ('conv4_1', ConvBlock(256, 512)),
                ('conv4_2', ConvBlock(512, 512)),
                ('maxpool4', nn.MaxPool2d((2, 2)))
            ])
        )

        # Encoder Block 5
        self.middle = nn.Sequential(
            OrderedDict([
                ('conv5_1', ConvBlock(512, 1024)),
                ('conv5_2', ConvBlock(1024, 1024))
            ])
        )

        self.upconv4 = nn.Sequential(OrderedDict([('upconv4', DeconvBlock(1024))]))

        # Decoder Block 4 
        self.decoder_4 = nn.Sequential(
            OrderedDict([
                ('deconv4_1', ConvBlock(512, 512)),
                ('deconv4_2', ConvBlock(512, 512))
            ])
        )

        self.upconv3 = nn.Sequential(OrderedDict([('upconv3', DeconvBlock(512))]))

        # Decoder Block 3
        self.decoder_3 = nn.Sequential(
            OrderedDict([
                ('deconv3_1', ConvBlock(256, 256)),
                ('deconv3_2', ConvBlock(256, 256))
            ])
        )

        self.upconv2 = nn.Sequential(OrderedDict([('upconv2', DeconvBlock(256))]))

        # Decoder Block 2
        self.decoder_2 = nn.Sequential(
            OrderedDict([
                ('deconv2_1', ConvBlock(128, 128)),
                ('deconv2_2', ConvBlock(128, 128))
            ])
        )

        self.upconv1 = nn.Sequential(OrderedDict([('upconv1', DeconvBlock(128))]))

        # Decoder Block 1
        self.decoder_1 = nn.Sequential(
            OrderedDict([
                ('deconv1_1', ConvBlock(64, 64)),
                ('deconv1_2', ConvBlock(64, 64)),
                ('deconv1_3', ConvBlock(64, self.n_classes, (1,1), activation=False)),
            ])
        )

    def forward(self, x):
        # Encoder
        # 1-64
        encoder_1 = self.encoder_1(x)
        # 64-128
        encoder_2 = self.encoder_2(encoder_1)
        # 128-256
        encoder_3 = self.encoder_2(encoder_2)
        # 256-512
        encoder_4 = self.encoder_2(encoder_3)
        
        # 512-1024
        x = self.encoder_2(encoder_4)

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

class UNet_ResNet(nn.Module):
    def __init__(self, in_channels, n_classes, dropout=0.5):
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
        encoder_3 = self.encoder_2(encoder_2)
        # 256-512
        encoder_4 = self.encoder_2(encoder_3)
        
        # 512-1024
        x = self.encoder_2(encoder_4)

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