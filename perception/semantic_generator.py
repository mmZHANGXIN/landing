"""
语义图生成器 - 将 HALSS BEV 安全评估结果转换为语义图
"""

import numpy as np
import cv2


class SemanticGenerator:
    """
    从 HALSS BEV 评估结果生成语义图 (128x128 uint8)
    兼容 RL 模型输入要求。
    """

    def __init__(self, cfg: dict):
        self.safe_id = cfg.get("safe_class_id", 1)
        self.danger_id = cfg.get("danger_class_id", 9)
        self.img_w = cfg.get("img_width", 128)
        self.img_h = cfg.get("img_height", 128)

    def generate(self, bev_result: dict) -> np.ndarray:
        """
        输入: HALSS 评估结果 dict
        输出: (img_h, img_w) uint8 语义图 (每个像素是类别ID)
        """
        if bev_result is None:
            return np.full((self.img_h, self.img_w), self.danger_id, dtype=np.uint8)

        safe_mesh = bev_result["safe_mesh"]  # (rows, cols) bool

        # 默认全部危险
        sem_mesh = np.full(safe_mesh.shape, self.danger_id, dtype=np.uint8)
        sem_mesh[safe_mesh] = self.safe_id

        # Resize 到 RL 尺寸 (最近邻，保持类别ID)
        sem_resized = cv2.resize(
            sem_mesh, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST
        )
        return sem_resized

    def colorize(self, sem_map: np.ndarray) -> np.ndarray:
        """语义图 → BGR 彩色可视化"""
        # 简化配色: 安全=绿色, 危险=红色, 未知=灰色
        h, w = sem_map.shape
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        vis[sem_map == self.safe_id] = [0, 255, 0]      # 绿色
        vis[sem_map == self.danger_id] = [0, 0, 255]     # 红色
        # 其他类别设为灰色
        mask_other = (sem_map != self.safe_id) & (sem_map != self.danger_id)
        vis[mask_other] = [128, 128, 128]
        return vis
