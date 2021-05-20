from model.config import N_CLASSES
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from utils.common import ConvBlock, DeconvBlock, DoubleConvBlock, ResidualBlock

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   


class UNet(nn.Module):
    def __init__(self, n_class=N_CLASSES):
        super().__init__()
                
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)
        
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out

class UNet1(nn.Module):
    def __init__(self, in_channels=3, n_classes=N_CLASSES):
        super(UNet, self).__init__()

        # Maxpool 
        self.pool = nn.MaxPool2d((2,2))

        # Up sampling
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Encoder 
        self.encoder_1 = DoubleConvBlock(in_channels, 64)
        self.encoder_2 = DoubleConvBlock(64, 128)
        self.encoder_3 = DoubleConvBlock(128, 256)
        self.encoder_4 = DoubleConvBlock(256, 512)

        # Middle
        self.middle = DoubleConvBlock(512, 1024)

        # Decoder
        self.decoder_4 = DoubleConvBlock(1024, 512)
        self.decoder_4 = DoubleConvBlock(512, 256)
        self.decoder_4 = DoubleConvBlock(256, 128)
        self.decoder_4 = DoubleConvBlock(128, 64)
        self.conv_last = nn.Conv2d(64, n_classes, (1, 1))

    def forward(self, x):
        # Encoder
        # 3-64
        encoder_1 = self.encoder_1(x)
        x = self.pool(encoder_1)
        
        # 64-128
        encoder_2 = self.encoder_2(x)
        x = self.pool(encoder_2)

        # 128-256
        encoder_3 = self.encoder_3(x)
        x = self.pool(encoder_3)
        
        # 256-512
        encoder_4 = self.encoder_4(x)
        x = self.pool(encoder_4)

        # Middle: 512-1024
        x = self.middle(x)

        # Decoder
        # 1024-512
        x = self.upsample(x)
        x = torch.cat([x, encoder_4], dim=1)
        x = self.decoder_4(x)

        # 512-256
        x = self.upsample(x)
        x = torch.cat([x, encoder_3], dim=1)
        x = self.decoder_3(x)

        # 256-128
        x = self.upsample(x)
        x = torch.cat([x, encoder_2], dim=1)
        x = self.decoder_2(x)

        #128-64-2
        x = self.upsample(x)
        x = torch.cat([x, encoder_1], dim=1)
        x = self.decoder_1(x)

        out = self.conv_last(x)
        return out

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