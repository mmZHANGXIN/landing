"""
语义类别定义与颜色映射
从 arch/FaultyYawLanding/DeepRL/semantics_classes.py 迁移
"""

from enum import IntEnum
from typing import Tuple, Dict
import numpy as np
import cv2


class Color:
    def __init__(self, r: int, g: int, b: int):
        self.r = r
        self.g = g
        self.b = b


class SemClasses(IntEnum):
    kUnknown = -1
    kPavement = 0
    kTerrain = 1
    kWater = 2
    kSky = 3
    kBuilding = 4
    kVegetation = 5
    kPerson = 6
    kRider = 7
    kVehicle = 8
    kOthers = 9


CLASS_TO_RGB: Dict[SemClasses, Tuple[int, int, int]] = {
    SemClasses.kUnknown:     (0, 0, 0),
    SemClasses.kPavement:    (81, 0, 81),
    SemClasses.kTerrain:     (152, 251, 152),
    SemClasses.kWater:       (150, 170, 250),
    SemClasses.kSky:         (70, 130, 180),
    SemClasses.kBuilding:    (70, 70, 70),
    SemClasses.kVegetation:  (107, 142, 35),
    SemClasses.kPerson:      (220, 20, 60),
    SemClasses.kRider:       (255, 0, 0),
    SemClasses.kVehicle:     (0, 0, 142),
    SemClasses.kOthers:      (250, 170, 30),
}


def get_color_semantic_by_id(sem_mask: np.ndarray) -> np.ndarray:
    """将类别ID掩码转换为RGB彩色图像"""
    H, W = sem_mask.shape
    color_img = np.zeros((H, W, 3), dtype=np.uint8)
    for class_id, rgb in CLASS_TO_RGB.items():
        if class_id >= 0:
            color_img[sem_mask == int(class_id)] = rgb
    return color_img
