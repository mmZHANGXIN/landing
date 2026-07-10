"""
全局安全先验评估 — 九宫格风险分析
从 arch/FaultyYawLanding/perception/coarse_perception.py 迁移适配

管线:
  GIS卫星图 → SegFormer 语义分割 → 9宫格滑动窗口 →
  风险值映射 → 选出最低风险格中心 → 输出安全着陆点GPS坐标

适配改动:
  1. 移除 RflySim VisionCaptureApi 依赖, 改用本地图像文件
  2. 输出 GPS 坐标而非像素坐标, 供 MAVSDK 位置控制使用
  3. 支持只加载预计算语义掩码 (跳过推理) 以加速
"""

import cv2
import numpy as np
import os
import time
import logging
import json

logger = logging.getLogger("GlobalSafetyPrior")


class GlobalSafetyPrior:
    """
    全局安全先验评估器。

    两种使用模式:
      模式A (有分割模型): 输入 RGB 卫星图, 内部运行 SegFormer 推理
      模式B (预计算掩码): 输入已保存的语义掩码 .png, 直接做九宫格分析
    """

    # 风险值映射: 类别ID → 风险分数 (0=最安全, 4=最危险)
    RISK_LUT = {
        1: 0.0,   # Terrain / 平坦草地 → 最安全
        0: 1.0,   # Pavement / 铺装路面
        5: 2.0,   # Vegetation / 植被
        4: 3.0,   # Building / 建筑
    }
    DEFAULT_RISK = 4.0  # 未知/其他类别

    def __init__(self, cfg: dict = None):
        """
        cfg 可包含:
          - model_arch: 'segformer' | 'fastscnn' | 'unet'
          - model_path: 分割权重路径
          - safe_class_id: 安全类别ID (默认1)
          - input_size: 分割输入尺寸 (默认512,512)
        """
        cfg = cfg or {}
        self.safe_class_id = cfg.get('safe_class_id', 1)
        self.seg_model = None

        # 如果提供了模型配置, 初始化分割模型
        model_arch = cfg.get('model_arch')
        model_path = cfg.get('model_path')
        if model_arch and model_path:
            self._init_seg_model(model_arch, model_path,
                                 cfg.get('input_size', (512, 512)))

    def _init_seg_model(self, model_arch: str, model_path: str,
                        input_size: tuple):
        """初始化语义分割模型"""
        import torch
        from .segmentation import SegmentationInference
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"[GlobalSafetyPrior] Loading {model_arch} on {device} ...")
        self.seg_model = SegmentationInference(
            model_arch=model_arch,
            model_path=model_path,
            num_classes=10,
            input_size=input_size,
            device=device,
        )

    # ==================================================================
    # 核心: 九宫格风险评估
    # ==================================================================

    def assess(self, rgb_img: np.ndarray = None,
               sem_mask: np.ndarray = None) -> dict:
        """
        执行全局安全评估。

        输入 (二选一):
          rgb_img:   RGB 卫星图 (H,W,3) — 需已初始化 seg_model
          sem_mask:  预计算语义掩码 (H,W) uint8 — 跳过推理

        返回:
          dict: {
            'best_center_px': (cx, cy),     # 最佳安全格中心像素坐标
            'best_center_gps': (lat, lon),  # 最佳安全格中心 GPS (需 georef)
            'min_risk': float,              # 最小风险值
            'risk_grid': np.ndarray (3,3),  # 九宫格风险矩阵
            'safe_mask': np.ndarray (H,W),  # 最佳安全格二值掩码
            'sem_mask': np.ndarray (H,W),   # 原始语义掩码
            'overlay_img': np.ndarray (H,W,3),  # 可视化叠加图
          }
        """
        # Step 1: 获取语义掩码
        if sem_mask is None:
            if rgb_img is None:
                raise ValueError("Must provide either rgb_img or sem_mask")
            if self.seg_model is None:
                raise RuntimeError("Seg model not initialized. Provide sem_mask or init with model config.")
            logger.info("[GlobalSafetyPrior] Running SegFormer inference...")
            sem_mask = self.seg_model.predict(rgb_img)  # (H,W) uint8
        else:
            rgb_img = np.zeros((sem_mask.shape[0], sem_mask.shape[1], 3), dtype=np.uint8)

        H, W = sem_mask.shape

        # Step 2: 生成可视化叠加图
        overlay_img = self._make_overlay(rgb_img, sem_mask)

        # Step 3: 构建风险矩阵
        risk_matrix = self._build_risk_matrix(sem_mask)

        # Step 4: 九宫格滑窗评估
        ch, cw = H // 3, W // 3
        risk_grid = np.zeros((3, 3), dtype=np.float32)
        min_risk = float('inf')
        best_center = (W // 2, H // 2)
        best_cell = (1, 1)
        best_cell_bounds = (0, 0, W, H)
        safe_mask = np.zeros_like(sem_mask, dtype=np.uint8)

        for row in range(3):
            for col in range(3):
                y0, y1 = row * ch, (row + 1) * ch if row < 2 else H
                x0, x1 = col * cw, (col + 1) * cw if col < 2 else W

                cell_risk = float(np.mean(risk_matrix[y0:y1, x0:x1]))
                risk_grid[row, col] = cell_risk

                # 绘制九宫格边框
                cv2.rectangle(overlay_img, (x0, y0), (x1, y1), (255, 255, 255), 1)
                text = f"{cell_risk:.2f}"
                cv2.putText(overlay_img, text, (x0 + 5, y0 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                cv2.putText(overlay_img, text, (x0 + 5, y0 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

                if cell_risk < min_risk:
                    min_risk = cell_risk
                    best_cell = (row, col)
                    best_cell_bounds = (x0, y0, x1, y1)
                    best_center = ((x0 + x1) // 2, (y0 + y1) // 2)
                    safe_mask.fill(0)
                    safe_mask[y0:y1, x0:x1] = 1

        # 标注最佳安全格中心
        cv2.circle(overlay_img, best_center, radius=6, color=(0, 0, 255), thickness=-1)
        cv2.circle(overlay_img, best_center, radius=8, color=(255, 255, 255), thickness=1)

        logger.info(f"[GlobalSafetyPrior] 九宫格完成. 最小风险={min_risk:.3f} "
                     f"@ 格({best_center[0]},{best_center[1]})")

        return {
            'best_center_px': best_center,
            'best_center_gps': None,  # 调用 set_georeference 后填充
            'best_cell': best_cell,
            'best_cell_bounds_px': best_cell_bounds,
            'min_risk': min_risk,
            'risk_grid': risk_grid,
            'safe_mask': safe_mask,
            'sem_mask': sem_mask,
            'overlay_img': overlay_img,
        }

    # ==================================================================
    # 辅助方法
    # ==================================================================

    def _build_risk_matrix(self, sem_mask: np.ndarray) -> np.ndarray:
        """类别ID掩码 → 风险值矩阵"""
        risk = np.full(sem_mask.shape, self.DEFAULT_RISK, dtype=np.float32)
        for class_id, risk_val in self.RISK_LUT.items():
            risk[sem_mask == class_id] = risk_val
        return risk

    def _make_overlay(self, rgb_img: np.ndarray,
                      sem_mask: np.ndarray) -> np.ndarray:
        """生成半透明语义叠加可视化图"""
        overlay = rgb_img.copy()
        try:
            from .segmentation.semantics_classes import get_color_semantic_by_id
            color_mask = get_color_semantic_by_id(sem_mask)
            if color_mask.shape[:2] != overlay.shape[:2]:
                color_mask = cv2.resize(color_mask,
                                        (overlay.shape[1], overlay.shape[0]),
                                        interpolation=cv2.INTER_NEAREST)
            overlay = cv2.addWeighted(overlay, 0.5, color_mask, 0.5, 0)
        except Exception:
            safe_color = np.zeros_like(overlay)
            safe_color[sem_mask == self.safe_class_id] = [0, 255, 0]
            overlay = cv2.addWeighted(overlay, 1.0, safe_color, 0.5, 0)
        return overlay

    # ==================================================================
    # 地理参考 — 像素坐标 → GPS
    # ==================================================================

    def set_georeference(self, bounds: tuple, img_size: tuple):
        """
        设置图像的地理参考信息。

        bounds: (lon_left, lat_bottom, lon_right, lat_top)
        img_size: (width_px, height_px)
        """
        self._lon_left, self._lat_bot = bounds[0], bounds[1]
        self._lon_right, self._lat_top = bounds[2], bounds[3]
        self._img_w, self._img_h = img_size

    def pixel_to_gps(self, px: int, py: int) -> tuple:
        """像素坐标 → GPS (lat, lon)"""
        if not hasattr(self, '_lon_left'):
            raise RuntimeError("Call set_georeference() first.")
        lon = self._lon_left + (px / self._img_w) * (self._lon_right - self._lon_left)
        lat = self._lat_top - (py / self._img_h) * (self._lat_top - self._lat_bot)
        return (lat, lon)

    # ==================================================================
    # 一键评估 (从 GIS 影像文件)
    # ==================================================================

    def assess_from_file(self, image_path: str, sem_mask_path: str = None,
                         bounds: tuple = None) -> dict:
        """
        从本地 GIS 影像文件执行完整评估。

        返回的 best_center_gps 是可直接用于 MAVSDK 位置控制的 GPS 坐标。
        bounds 缺失时拒绝生成 GPS 目标点 (best_center_gps=None)。
        """
        if sem_mask_path and os.path.exists(sem_mask_path):
            sem_mask = cv2.imread(sem_mask_path, cv2.IMREAD_GRAYSCALE)
            rgb_img = cv2.imread(image_path)
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB) if rgb_img is not None else None
            result = self.assess(sem_mask=sem_mask)
        else:
            rgb_img = cv2.imread(image_path)
            if rgb_img is None:
                raise FileNotFoundError(f"Image not found: {image_path}")
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            result = self.assess(rgb_img=rgb_img)

        # 填充 GPS 坐标
        H, W = result['sem_mask'].shape
        result['source_image_path'] = image_path
        result['source_sem_mask_path'] = sem_mask_path
        result['segmentation_source'] = (
            "precomputed_mask" if sem_mask_path and os.path.exists(sem_mask_path)
            else "segmentation_model"
        )
        result['bounds'] = bounds
        result['image_size_px'] = (W, H)
        if bounds is None:
            logger.warning("[GlobalSafetyPrior] ⚠ No bounds provided — "
                           "best_center_gps NOT set. Cannot generate landing target.")
            # 明确拒绝: GPS 为空, 外部应据此拒绝进入 goto 阶段
            result['best_center_gps'] = None
        else:
            self.set_georeference(bounds, (W, H))
            cx, cy = result['best_center_px']
            result['best_center_gps'] = self.pixel_to_gps(cx, cy)
            logger.info(f"[GlobalSafetyPrior] Safe landing GPS: "
                         f"lat={result['best_center_gps'][0]:.6f}, "
                         f"lon={result['best_center_gps'][1]:.6f}")

        return result

    def save_results(self, result: dict, output_dir: str = "./gis_data"):
        """保存评估结果到磁盘"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time() * 1000)

        cv2.imwrite(os.path.join(output_dir, f"overlay_{timestamp}.png"),
                    cv2.cvtColor(result['overlay_img'], cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, f"safe_mask_{timestamp}.png"),
                    result['safe_mask'] * 255)
        cv2.imwrite(os.path.join(output_dir, f"sem_mask_{timestamp}.png"),
                    result['sem_mask'])
        summary = {
            "timestamp_ms": timestamp,
            "source_image_path": _to_json_value(result.get("source_image_path")),
            "source_sem_mask_path": _to_json_value(result.get("source_sem_mask_path")),
            "segmentation_source": _to_json_value(result.get("segmentation_source")),
            "bounds": _to_json_value(result.get("bounds")),
            "image_size_px": _to_json_value(result.get("image_size_px")),
            "best_cell": _to_json_value(result.get("best_cell")),
            "best_cell_bounds_px": _to_json_value(result.get("best_cell_bounds_px")),
            "best_center_px": _to_json_value(result.get("best_center_px")),
            "best_center_gps": _to_json_value(result.get("best_center_gps")),
            "min_risk": _to_json_value(result.get("min_risk")),
            "risk_grid": _to_json_value(result.get("risk_grid")),
            "has_gps_target": result.get("best_center_gps") is not None,
        }
        with open(os.path.join(output_dir, f"global_prior_{timestamp}.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"[GlobalSafetyPrior] Results saved to {output_dir}")


def _to_json_value(value):
    if value is None:
        return None
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value
