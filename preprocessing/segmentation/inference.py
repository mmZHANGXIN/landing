"""
语义分割推理封装
从 arch/FaultyYawLanding/perception/segmentation/inference.py 迁移
"""

import os
import torch
import cv2
import numpy as np
from .segmentation_factory import ModelFactory


class SegmentationInference:
    """
    语义分割推理器，支持 SegFormer / FastSCNN / UNet。
    """

    def __init__(self, model_arch, model_path, num_classes=10, input_size=(512, 512), device=None):
        self.device = torch.device(device) if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True

        self.input_size = input_size

        # 加载模型
        self.model = ModelFactory.get_model(model_arch, num_classes=num_classes)
        self.model.to(self.device)

        if model_path and os.path.exists(model_path):
            print(f"[SegInference] Loading weights from {model_path} ...")
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
        else:
            print(f"[SegInference] WARNING: checkpoint not found at {model_path}")

        self.model.eval()

        # ImageNet 归一化参数
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def predict(self, img):
        """
        输入: RGB 图像 (H, W, 3) uint8
        输出: 语义掩码 (H, W) uint8, 类别ID 0-9
        """
        original_h, original_w = img.shape[:2]

        # 预处理
        img_resized = cv2.resize(img, self.input_size)
        img_norm = (img_resized.astype(np.float32) / 255.0 - self.mean) / self.std
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)

        # 推理
        with torch.no_grad():
            output = self.model(img_tensor)
            if isinstance(output, (tuple, list)):
                output = output[0]
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # 恢复原分辨率
        if (original_h, original_w) != self.input_size:
            pred_mask = cv2.resize(pred_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

        return pred_mask
