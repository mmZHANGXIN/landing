#!/usr/bin/env python3
"""
PyTorch 感知发布节点 (orin_perception_pytorch)
==============================================
职责:
  1. 接收 LiDAR 点云 / RGB / 位姿
  2. 点云投影 → 稀疏深度图 + valid_mask
  3. 语义分割 → semantic_id 图
  4. 通过 ZeroMQ PUB 发布 sparse_depth_frame

环境: PyTorch + OpenCV + HALSS Bayesian UNet / FastSCNN
"""

import os
import sys
import time
import logging
import argparse
from typing import Optional, Dict, Any

import numpy as np
import cv2

try:
    import zmq
    HAS_ZMQ = True
except ImportError:
    zmq = None
    HAS_ZMQ = False

# ---- 项目内模块 ----
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control.zmq_protocol import (
    serialize_sparse_depth_frame,
    MSG_TYPE_SPARSE_DEPTH,
    DTYPE_SPARSE_DEPTH,
    DTYPE_VALID_MASK,
    DTYPE_SEMANTIC_ID,
)
from perception.depth_projection import DepthProjector
from perception.halss_bayesian import HalssBayesian

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("PerceptionPublisher")


class PerceptionPublisher:
    """
    感知发布节点。
    
    输入源可以是:
      - ROS2 topic (Mid360 + FAST-LIO pose)
      - 离线 bag 回放
      - 测试模式: 从文件加载点云/RGB
    """

    def __init__(
        self,
        pub_address: str = "tcp://127.0.0.1:5555",
        img_width: int = 752,
        img_height: int = 480,
        out_width: int = 128,
        out_height: int = 128,
        dmax: float = 30.0,
        halss_weight_path: Optional[str] = None,
        halss_grid_res: int = 64,
        mc_samples: int = 5,
        uncertainty_threshold: float = 0.3,
        safe_class_id: int = 1,
        danger_class_id: int = 9,
        lidar_position_body_m: tuple = (0.13, 0.0, 0.08),
        lidar_yaw_offset_deg: float = 0.0,
        lidar_pitch_down_deg: float = 26.0,
        halss_roi_radius_body: float = 25.0,
        halss_min_down_m: float = 0.05,
        halss_max_down_m: float = 30.0,
        halss_yaw_only: bool = True,
        source_mode: str = "offline",
        bag_path: Optional[str] = None,
    ):
        if not HAS_ZMQ:
            raise ImportError(
                "pyzmq is required for PerceptionPublisher. "
                "Install with: pip install pyzmq"
            )
        # ---- ZeroMQ ----
        self.ctx = zmq.Context()
        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.bind(pub_address)
        logger.info(f"Perception PUB bound to {pub_address}")

        # ---- 参数 ----
        self.img_w = img_width
        self.img_h = img_height
        self.out_w = out_width
        self.out_h = out_height
        self.dmax = dmax
        self.frame_id = 0
        self.source_mode = source_mode
        self.bag_path = bag_path

        # ---- 深度投影器 (PyTorch CUDA) ----
        self.projector = DepthProjector(
            img_width=out_width,
            img_height=out_height,
            max_range=dmax,
            backend="torch_cuda",
            mode="bev",  # Body-frame ROI projection
        )

        # ---- 语义分割 (HALSS Bayesian UNet) ----
        self.halss = None
        if halss_weight_path and os.path.exists(halss_weight_path):
            self.halss = HalssBayesian(
                weight_path=halss_weight_path,
                grid_res=halss_grid_res,
                mc_samples=mc_samples,
                uncertainty_threshold=uncertainty_threshold,
                safe_class_id=safe_class_id,
                danger_class_id=danger_class_id,
            )
            logger.info(f"HALSS Bayesian UNet loaded from {halss_weight_path}")
        else:
            logger.warning("No HALSS weight path provided; semantic output will be dummy")

        self.halss_grid_res = halss_grid_res
        self.lidar_pos_body = np.array(lidar_position_body_m, dtype=np.float32)
        self.lidar_yaw_offset = np.deg2rad(lidar_yaw_offset_deg)
        self.lidar_pitch_down = np.deg2rad(lidar_pitch_down_deg)
        self.halss_roi_radius = halss_roi_radius_body
        self.halss_min_down = halss_min_down_m
        self.halss_max_down = halss_max_down_m
        self.halss_yaw_only = halss_yaw_only

        # ---- 统计 ----
        self.last_pub_time = time.time()
        self.pub_count = 0
        self.fps = 0.0

    # ================================================================
    # 核心管线
    # ================================================================

    def process_frame(
        self,
        points_world: Optional[np.ndarray] = None,
        points_body: Optional[np.ndarray] = None,
        drone_pose: Optional[np.ndarray] = None,
        rgb_image: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        处理单帧传感器数据 → 返回发布消息所需的字段.

        参数:
          points_world: (N, 3) 世界系点云
          points_body:  (N, 3) 机体系点云 (优先使用, 跳过坐标变换)
          drone_pose:   (7,) [x, y, z, qw, qx, qy, qz] 世界系位姿
          rgb_image:    (H, W, 3) RGB/BGR 图像 (可选)

        返回:
          dict: sparse_depth, valid_mask, semantic_id, pose, intrinsics, depth_scale
        """
        # ---- 1. 深度投影 ----
        if points_body is not None and len(points_body) > 0:
            # 机体系直接投影 (HALSS-aligned body ROI)
            sparse_depth = self.projector.project_body_roi(
                points_body, source_shape=(self.halss_grid_res, self.halss_grid_res)
            )
            valid_mask = (sparse_depth > 0.01).astype(np.uint8) & \
                         (sparse_depth < self.dmax).astype(np.uint8)
        elif points_world is not None and len(points_world) > 0 and drone_pose is not None:
            # 世界系 → 透视投影
            sparse_depth = self.projector.project(points_world, drone_pose)
            valid_mask = (sparse_depth > 0.01).astype(np.uint8) & \
                         (sparse_depth < self.dmax).astype(np.uint8)
        else:
            # 无点云 → 全零
            sparse_depth = np.zeros((self.out_h, self.out_w), dtype=np.float32)
            valid_mask = np.zeros((self.out_h, self.out_w), dtype=np.uint8)

        # ---- 2. 语义分割 ----
        if self.halss is not None and points_body is not None and len(points_body) > 0:
            semantic_id = self._run_halss_semantic(points_body)
        elif rgb_image is not None:
            # 从 RGB 做语义分割的 fallback (如果有加载 FastSCNN 模型)
            semantic_id = self._run_rgb_semantic(rgb_image)
        else:
            # 无语义 → 全 Unknown
            semantic_id = np.full((self.out_h, self.out_w), 255, dtype=np.uint8)

        # ---- 3. 组装发布数据 ----
        pose_list = None
        if drone_pose is not None:
            pose_list = drone_pose.tolist() if hasattr(drone_pose, "tolist") else list(drone_pose)

        intrinsics = {
            "fx": self.projector.fx,
            "fy": self.projector.fy,
            "cx": self.projector.cx,
            "cy": self.projector.cy,
        }

        return {
            "sparse_depth": sparse_depth,
            "valid_mask": valid_mask,
            "semantic_id": semantic_id,
            "pose": pose_list,
            "camera_intrinsics": intrinsics,
            "depth_scale": 1.0,
        }

    def publish(self, data: Dict[str, Any]):
        """序列化并发布一帧."""
        header, payload = serialize_sparse_depth_frame(
            frame_id=self.frame_id,
            sparse_depth=data["sparse_depth"],
            valid_mask=data["valid_mask"],
            semantic_id=data["semantic_id"],
            pose=np.array(data["pose"]) if data.get("pose") else None,
            camera_intrinsics=data.get("camera_intrinsics"),
            depth_scale=data.get("depth_scale", 1.0),
            compress=False,
        )
        self.pub.send(header + payload)
        self.frame_id += 1

        # FPS 统计
        now = time.time()
        self.pub_count += 1
        if now - self.last_pub_time >= 1.0:
            self.fps = self.pub_count / (now - self.last_pub_time)
            self.pub_count = 0
            self.last_pub_time = now
            logger.debug(f"Publishing at {self.fps:.1f} FPS, frame_id={self.frame_id}")

    # ================================================================
    # 语义分割内部方法
    # ================================================================

    def _run_halss_semantic(self, points_body: np.ndarray) -> np.ndarray:
        """运行 HALSS Bayesian UNet 语义分割, 返回 semantic_id (HxW uint8)."""
        try:
            result = self.halss.process(
                points_body,
                lidar_pos_body=self.lidar_pos_body,
                lidar_yaw_offset=self.lidar_yaw_offset,
                lidar_pitch_down=self.lidar_pitch_down,
                roi_radius=self.halss_roi_radius,
                min_down_m=self.halss_min_down,
                max_down_m=self.halss_max_down,
                yaw_only=self.halss_yaw_only,
            )
            # result 包含 "safe_mesh" (bool grid) 和 "semantic_map"
            safe_mesh = result.get("safe_mesh")
            if safe_mesh is not None:
                # safe_mesh: True=safe (class 1), False=danger (class 9)
                sem = np.full(safe_mesh.shape, 9, dtype=np.uint8)  # default danger
                sem[safe_mesh] = 1  # safe class
                if sem.shape != (self.out_h, self.out_w):
                    sem = cv2.resize(sem, (self.out_w, self.out_h),
                                     interpolation=cv2.INTER_NEAREST)
                return sem
        except Exception as e:
            logger.error(f"HALSS semantic failed: {e}")

        return np.full((self.out_h, self.out_w), 255, dtype=np.uint8)

    def _run_rgb_semantic(self, rgb_image: np.ndarray) -> np.ndarray:
        """从 RGB 图像运行语义分割 (placeholder)."""
        # 此处可接入 FastSCNN 或其他 PyTorch 语义分割模型
        # 当前返回全 Unknown
        logger.warning("RGB semantic not implemented; returning unknown")
        sem = np.full((self.out_h, self.out_w), 255, dtype=np.uint8)
        return sem

    # ================================================================
    # 运行循环
    # ================================================================

    def run_once_offline(self, points_body: np.ndarray):
        """离线单帧测试."""
        data = self.process_frame(points_body=points_body)
        self.publish(data)
        logger.info(f"Published frame {self.frame_id - 1}: "
                    f"depth range [{data['sparse_depth'].min():.2f}, {data['sparse_depth'].max():.2f}] m, "
                    f"valid pixels {data['valid_mask'].sum()}/{data['valid_mask'].size}")

    def run_loop(self, data_source):
        """
        主循环: 从 data_source 迭代器获取数据并发布.

        data_source 应为 generator, yield dict:
          {"points_body": ..., "drone_pose": ..., "rgb": ...}
        """
        logger.info("Starting perception publish loop...")
        try:
            for sensor_data in data_source:
                data = self.process_frame(
                    points_body=sensor_data.get("points_body"),
                    points_world=sensor_data.get("points_world"),
                    drone_pose=sensor_data.get("drone_pose"),
                    rgb_image=sensor_data.get("rgb"),
                )
                self.publish(data)
        except KeyboardInterrupt:
            logger.info("Perception publisher interrupted.")
        finally:
            self.pub.close()
            self.ctx.term()

    def close(self):
        self.pub.close()
        self.ctx.term()


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Perception Publisher (PyTorch)")
    parser.add_argument("--pub-address", default="tcp://127.0.0.1:5555",
                        help="ZeroMQ PUB address")
    parser.add_argument("--halss-weight", default=None,
                        help="Path to HALSS Bayesian UNet weights (.pth)")
    parser.add_argument("--mode", default="offline", choices=["offline", "ros2", "bag"],
                        help="Data source mode")
    parser.add_argument("--test-npz", default=None,
                        help="Path to .npz file with points_body for offline test")
    args = parser.parse_args()

    pub = PerceptionPublisher(
        pub_address=args.pub_address,
        halss_weight_path=args.halss_weight,
    )

    if args.mode == "offline" and args.test_npz:
        data = np.load(args.test_npz)
        points_body = data.get("points_body")
        if points_body is not None:
            pub.run_once_offline(points_body)
        else:
            logger.error("No 'points_body' key in .npz file")
    else:
        logger.info("PerceptionPublisher initialized. Use run_loop() with a data source.")

    pub.close()


if __name__ == "__main__":
    main()
