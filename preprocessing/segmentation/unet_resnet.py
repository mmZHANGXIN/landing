"""
UNet-ResNet18 语义分割模型
从 arch/FaultyYawLanding/perception/segmentation/unet_resnet.py 迁移
"""

import torch
import torch.nn as nn
from torchvision import models


class UNetResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.base = models.resnet18(pretrained=True)
        self.enc0 = nn.Sequential(self.base.conv1, self.base.bn1, self.base.relu)
        self.enc1 = nn.Sequential(self.base.maxpool, self.base.layer1)
        self.enc2 = self.base.layer2
        self.enc3 = self.base.layer3
        self.enc4 = self.base.layer4

        self.up1 = DecoderBlock(512, 256)
        self.up2 = DecoderBlock(256, 128)
        self.up3 = DecoderBlock(128, 64)
        self.up4 = DecoderBlock(64, 64)
        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d1 = self.up1(e4, e3)
        d2 = self.up2(d1, e2)
        d3 = self.up3(d2, e1)
        d4 = self.up4(d3, e0)

        out = self.final(d4)
        return torch.nn.functional.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, 3, 1, 1), nn.BatchNorm2d(out_ch), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1), nn.BatchNorm2d(out_ch), nn.ReLU()
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if x1.size() != x2.size():
            x1 = torch.nn.functional.interpolate(x1, size=x2.shape[2:])
        return self.conv(torch.cat([x2, x1], dim=1))
