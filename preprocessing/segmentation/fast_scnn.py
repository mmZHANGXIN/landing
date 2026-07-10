"""
FastSCNN 语义分割模型 (轻量级, 适合嵌入式)
从 arch/FaultyYawLanding/perception/segmentation/fast_scnn.py 迁移
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FastSCNN(nn.Module):
    def __init__(self, num_classes=10, aux=False, **kwargs):
        super(FastSCNN, self).__init__()
        self.aux = aux
        self.learning_to_downsample = LearningToDownsample(32, 48, 64)
        self.global_feature_extractor = GlobalFeatureExtractor(
            in_channels=64, block_channels=[64, 96, 128],
            out_channels=128, t=6, num_blocks=[3, 3, 3]
        )
        self.feature_fusion = FeatureFusionModule(64, 128, 128)
        self.classifier = Classifer(128, num_classes)
        if self.aux:
            self.aux_classifier = Classifer(64, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        size = x.size()[2:]
        higher_res_features = self.learning_to_downsample(x)
        x = self.global_feature_extractor(higher_res_features)
        x = self.feature_fusion(higher_res_features, x)
        x = self.classifier(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        return x


class _ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class _DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(_DSConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class LearningToDownsample(nn.Module):
    def __init__(self, dw_channels1=32, dw_channels2=48, out_channels=64):
        super(LearningToDownsample, self).__init__()
        self.conv = _ConvBNReLU(3, dw_channels1, 3, 2)
        self.dsconv1 = _DSConv(dw_channels1, dw_channels2, 2)
        self.dsconv2 = _DSConv(dw_channels2, out_channels, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, t=6, stride=1):
        super(InvertedResidual, self).__init__()
        hidden_dim = in_channels * t
        self.use_residual = stride == 1 and in_channels == out_channels
        layers = []
        if t != 1:
            layers.append(_ConvBNReLU(in_channels, hidden_dim, 1, 1, padding=0))
        layers.extend([
            _ConvBNReLU(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class GlobalFeatureExtractor(nn.Module):
    def __init__(self, in_channels=64, block_channels=[64, 96, 128],
                 out_channels=128, t=6, num_blocks=[3, 3, 3]):
        super(GlobalFeatureExtractor, self).__init__()
        self.bottleneck1 = self._make_layer(InvertedResidual, in_channels, block_channels[0], num_blocks[0], t, 2)
        self.bottleneck2 = self._make_layer(InvertedResidual, block_channels[0], block_channels[1], num_blocks[1], t, 2)
        self.bottleneck3 = self._make_layer(InvertedResidual, block_channels[1], block_channels[2], num_blocks[2], t, 1)
        self.ppm = PyramidPooling(block_channels[2], out_channels)

    def _make_layer(self, block, inplanes, planes, blocks, t=6, stride=1):
        layers = [block(inplanes, planes, t, stride)]
        for _ in range(1, blocks):
            layers.append(block(planes, planes, t, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.ppm(x)
        return x


class PyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PyramidPooling, self).__init__()
        inter_channels = in_channels // 4
        self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                   nn.BatchNorm2d(inter_channels), nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.AdaptiveAvgPool2d(2), nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                   nn.BatchNorm2d(inter_channels), nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.AdaptiveAvgPool2d(3), nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                   nn.BatchNorm2d(inter_channels), nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.AdaptiveAvgPool2d(6), nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                   nn.BatchNorm2d(inter_channels), nn.ReLU(True))
        self.out = nn.Sequential(
            nn.Conv2d(in_channels + inter_channels * 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(True)
        )

    def forward(self, x):
        h, w = x.size()[2:]
        feat1 = F.interpolate(self.conv1(x), (h, w), mode='bilinear', align_corners=True)
        feat2 = F.interpolate(self.conv2(x), (h, w), mode='bilinear', align_corners=True)
        feat3 = F.interpolate(self.conv3(x), (h, w), mode='bilinear', align_corners=True)
        feat4 = F.interpolate(self.conv4(x), (h, w), mode='bilinear', align_corners=True)
        return self.out(torch.cat([x, feat1, feat2, feat3, feat4], 1))


class FeatureFusionModule(nn.Module):
    def __init__(self, high_in_channels, low_in_channels, out_channels):
        super(FeatureFusionModule, self).__init__()
        self.dwconv = _DSConv(low_in_channels, out_channels, 1)
        self.conv_high = nn.Sequential(
            nn.Conv2d(high_in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(True)

    def forward(self, high_feat, low_feat):
        low_feat = self.dwconv(low_feat)
        high_feat = F.interpolate(self.conv_high(high_feat), low_feat.size()[2:],
                                  mode='bilinear', align_corners=True)
        return self.relu(low_feat + high_feat)


class Classifer(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Classifer, self).__init__()
        self.dsconv1 = _DSConv(in_channels, in_channels, 1)
        self.dsconv2 = _DSConv(in_channels, in_channels, 1)
        self.conv = nn.Conv2d(in_channels, num_classes, 1)

    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = F.dropout(x, 0.1, training=self.training)
        return self.conv(x)
