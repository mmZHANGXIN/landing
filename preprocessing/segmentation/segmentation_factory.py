"""
模型工厂 — 根据架构名返回对应模型
"""

import torch
from .fast_scnn import FastSCNN
from .segFormer import SegFormer
from .unet_resnet import UNetResNet18


class ModelFactory:
    @staticmethod
    def get_model(name, num_classes=10):
        name = name.lower()
        if name == 'fastscnn':
            return FastSCNN(num_classes=num_classes)
        elif name.startswith('segformer'):
            variant = 'b0'
            if '_' in name:
                variant = name.split('_')[1]
            return SegFormer(num_classes=num_classes, variant=variant, pretrained=False)
        elif name == 'unet':
            return UNetResNet18(num_classes=num_classes)
        else:
            raise ValueError(f"Unknown model: {name}")
