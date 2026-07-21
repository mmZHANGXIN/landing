#!/usr/bin/env python3
"""
TF/Keras 深度补全服务 (orin_depth_tfkeras)
===========================================
职责:
  1. 订阅 ZeroMQ sparse_depth_frame (来自 PyTorch 感知)
  2. 运行 Sparsity-Invariant CNN 深度补全
  3. 发布 ZeroMQ dense_depth_frame (送往 TF1 DRL)

环境: tensorflow>=2.x (tf.keras), Sparsity-Invariant-CNNs-master

注意:
  - 本服务独立于 DeepRL TF1 环境运行, 避免 tf.keras 与 stable-baselines TF1 冲突
  - 使用 PUB/SUB 模式, 只处理最新帧 (丢弃旧帧)
"""

import os
import sys
import time
import logging
import argparse
import threading
from typing import Optional, Dict, Any

import numpy as np

# zmq 是运行时依赖, 不在所有环境安装 — 延迟导入
try:
    import zmq
    HAS_ZMQ = True
except ImportError:
    zmq = None
    HAS_ZMQ = False

# ---- 项目内模块 ----
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control.zmq_protocol import (
    deserialize_sparse_depth_frame,
    serialize_dense_depth_frame,
    MSG_TYPE_SPARSE_DEPTH,
    HEADER_SIZE,
    unpack_header,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("DepthCompletion")


class SparseNetKeras:
    """
    Sparsity-Invariant CNN 模型包装器 (tf.keras).
    
    从 Sparsity-Invariant-CNNs-master 加载 .ckpt 权重。
    如果 tf.keras 不可用, 退化为最近邻上采样 (placeholder)。
    """

    def __init__(self, weight_path: Optional[str] = None,
                 input_size: int = 128, dmax: float = 30.0):
        self.input_size = input_size
        self.dmax = dmax
        self.model = None
        self._tf_available = False

        try:
            import tensorflow as tf
            self.tf = tf
            self._tf_available = True
            logger.info(f"TensorFlow version: {tf.__version__}")

            if weight_path and os.path.exists(weight_path + ".index"):
                self.model = self._build_model()
                self._load_weights(weight_path)
                logger.info(f"SparseNet loaded from {weight_path}")
            elif weight_path:
                logger.warning(f"Weight file not found: {weight_path}, using fallback")
            else:
                logger.warning("No weight path provided, using nearest-neighbor fallback")
        except ImportError:
            logger.warning("TensorFlow not available; using nearest-neighbor fallback")

    def _build_model(self):
        """
        构建 SparseNet 模型 (参考 Sparsity-Invariant-CNNs-master).
        
        此处为结构占位符; 实际使用时替换为源仓库提供的 model_fn。
        """
        tf = self.tf
        
        # 简化版 SparseNet 结构 (需与训练时匹配)
        inputs = tf.keras.layers.Input(
            shape=(self.input_size, self.input_size, 1), name="sparse_depth"
        )
        
        # Encoder
        x = tf.keras.layers.Conv2D(32, 7, strides=2, padding="same",
                                    activation="relu")(inputs)
        x = tf.keras.layers.Conv2D(64, 5, strides=2, padding="same",
                                    activation="relu")(x)
        x = tf.keras.layers.Conv2D(128, 3, strides=2, padding="same",
                                    activation="relu")(x)
        x = tf.keras.layers.Conv2D(256, 3, strides=2, padding="same",
                                    activation="relu")(x)
        
        # Decoder
        x = tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding="same",
                                             activation="relu")(x)
        x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same",
                                             activation="relu")(x)
        x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same",
                                             activation="relu")(x)
        x = tf.keras.layers.Conv2DTranspose(16, 3, strides=2, padding="same",
                                             activation="relu")(x)
        
        outputs = tf.keras.layers.Conv2D(1, 3, padding="same",
                                          activation="linear",
                                          name="dense_depth")(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def _load_weights(self, weight_path: str):
        """从 .ckpt 加载权重."""
        if self.model is not None:
            try:
                self.model.load_weights(weight_path)
                logger.info(f"Loaded weights from {weight_path}")
            except Exception as e:
                logger.error(f"Failed to load weights: {e}")
                logger.warning("Continuing with random weights — results will be invalid!")

    def predict(self, sparse_depth: np.ndarray,
                valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        输入稀疏深度 (HxW float32, 米) → 输出稠密深度 (128x128 float32, 米).

        如果 TF 不可用, 退化为最近邻填充。
        """
        import cv2

        # Resize 到模型输入尺寸
        h, w = sparse_depth.shape[:2]
        if (h, w) != (self.input_size, self.input_size):
            sd = cv2.resize(sparse_depth, (self.input_size, self.input_size),
                            interpolation=cv2.INTER_NEAREST)
            if valid_mask is not None:
                vm = cv2.resize(valid_mask.astype(np.float32),
                                (self.input_size, self.input_size),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
            else:
                vm = sd > 0.01
        else:
            sd = sparse_depth
            vm = valid_mask if valid_mask is not None else (sd > 0.01)

        if self._tf_available and self.model is not None:
            return self._predict_tf(sd, vm)
        else:
            return self._predict_fallback(sd, vm)

    def _predict_tf(self, sparse_depth: np.ndarray,
                    valid_mask: np.ndarray) -> np.ndarray:
        """TF/Keras 推理."""
        tf = self.tf

        # 归一化输入: 0~dmax → 0~1
        sd_norm = np.clip(sparse_depth, 0.0, self.dmax) / self.dmax
        sd_norm = sd_norm.astype(np.float32)[np.newaxis, ..., np.newaxis]  # (1, H, W, 1)

        output = self.model.predict(sd_norm, verbose=0)
        dense = output[0, ..., 0] * self.dmax  # 反归一化
        return dense.astype(np.float32)

    def _predict_fallback(self, sparse_depth: np.ndarray,
                          valid_mask: np.ndarray) -> np.ndarray:
        """最近邻填充回退 (无 TF 时)."""
        import cv2

        sd = sparse_depth.copy()
        sd[~valid_mask] = 0.0

        # 最近邻膨胀填充
        kernel = np.ones((5, 5), np.uint8)
        mask_dilated = cv2.dilate(valid_mask.astype(np.uint8), kernel, iterations=3)
        
        result = sd.copy()
        # 对空洞区域做中值填充
        for _ in range(3):
            result = cv2.medianBlur(result.astype(np.float32), 5)
        
        result = np.where(mask_dilated, result, self.dmax)
        return result.astype(np.float32)


class DepthCompletionService:
    """
    深度补全服务: SUB sparse_depth → SparseNet → PUB dense_depth
    """

    def __init__(
        self,
        sub_address: str = "tcp://127.0.0.1:5555",
        pub_address: str = "tcp://127.0.0.1:5556",
        sparsenet_weight: Optional[str] = None,
        input_size: int = 128,
        dmax: float = 30.0,
        high_pass_noop: bool = False,  # 调试: 跳过补全, 直接转发
    ):
        if not HAS_ZMQ:
            raise ImportError(
                "pyzmq is required for DepthCompletionService. "
                "Install with: pip install pyzmq"
            )
        # ---- ZeroMQ ----
        self.ctx = zmq.Context()
        
        # SUB socket: 接收稀疏深度
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.connect(sub_address)
        self.sub.setsockopt(zmq.SUBSCRIBE, b"")  # 订阅所有
        self.sub.setsockopt(zmq.CONFLATE, 1)      # 只保留最新消息
        
        # PUB socket: 发布稠密深度
        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.bind(pub_address)
        
        logger.info(f"DepthCompletion: SUB ← {sub_address}, PUB → {pub_address}")

        # ---- 模型 ----
        self.sparsenet = SparseNetKeras(
            weight_path=sparsenet_weight,
            input_size=input_size,
            dmax=dmax,
        )
        self.input_size = input_size
        self.dmax = dmax
        self.high_pass_noop = high_pass_noop

        # ---- 统计 ----
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0
        self.total_latency_ms = 0.0

    def run(self):
        """主循环."""
        logger.info("DepthCompletionService running...")
        poller = zmq.Poller()
        poller.register(self.sub, zmq.POLLIN)

        while True:
            try:
                socks = dict(poller.poll(timeout=100))  # 100ms timeout
                if self.sub in socks:
                    self._handle_frame()
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)

        self.cleanup()

    def _handle_frame(self):
        """处理一帧输入."""
        recv_time = time.time()
        message = self.sub.recv()

        if len(message) < HEADER_SIZE:
            logger.warning(f"Received undersized message: {len(message)} bytes")
            return

        # 解包稀疏深度帧
        try:
            data = deserialize_sparse_depth_frame(
                message[:HEADER_SIZE], message[HEADER_SIZE:]
            )
        except Exception as e:
            logger.error(f"Failed to deserialize: {e}")
            return

        sparse_depth = data["sparse_depth"]
        valid_mask = data["valid_mask"]
        semantic_id = data["semantic_id"]
        frame_id = data["frame_id"]

        # 深度补全
        if self.high_pass_noop:
            # 调试模式: 直接转发稀疏深度
            dense_depth = sparse_depth.astype(np.float32)
        else:
            dense_depth = self.sparsenet.predict(sparse_depth, valid_mask)

        # 确保输出尺寸
        if dense_depth.shape[:2] != (self.input_size, self.input_size):
            import cv2
            dense_depth = cv2.resize(dense_depth, (self.input_size, self.input_size),
                                     interpolation=cv2.INTER_LINEAR)

        # 发布
        header, payload = serialize_dense_depth_frame(
            frame_id=frame_id,
            dense_depth=dense_depth,
            semantic_id=semantic_id,
            compress=False,
        )
        self.pub.send(header + payload)

        # 统计
        latency_ms = (time.time() - recv_time) * 1000
        self.frame_count += 1
        now = time.time()
        if now - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (now - self.last_fps_time)
            logger.info(f"[DepthCompletion] FPS: {self.fps:.1f}, "
                        f"latency: {latency_ms:.1f}ms, "
                        f"depth_range: [{dense_depth.min():.2f}, {dense_depth.max():.2f}]m")
            self.frame_count = 0
            self.last_fps_time = now

    def run_once_offline(self, sparse_depth: np.ndarray,
                         valid_mask: np.ndarray,
                         semantic_id: np.ndarray) -> np.ndarray:
        """离线单帧测试."""
        dense_depth = self.sparsenet.predict(sparse_depth, valid_mask)
        logger.info(f"Offline test: dense_depth shape={dense_depth.shape}, "
                    f"range=[{dense_depth.min():.2f}, {dense_depth.max():.2f}]m")
        return dense_depth

    def cleanup(self):
        logger.info("Shutting down DepthCompletionService...")
        self.sub.close()
        self.pub.close()
        self.ctx.term()


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Depth Completion Service (TF/Keras)")
    parser.add_argument("--sub-address", default="tcp://127.0.0.1:5555",
                        help="SUB address (from perception)")
    parser.add_argument("--pub-address", default="tcp://127.0.0.1:5556",
                        help="PUB address (to DRL)")
    parser.add_argument("--sparsenet-weight", default=None,
                        help="Path to Sparsity-Invariant CNN .ckpt")
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--dmax", type=float, default=30.0)
    parser.add_argument("--high-pass-noop", action="store_true",
                        help="Skip completion, forward sparse as dense (debug)")
    parser.add_argument("--offline-test", default=None,
                        help="Path to .npz with sparse_depth/valid_mask for offline test")
    args = parser.parse_args()

    service = DepthCompletionService(
        sub_address=args.sub_address,
        pub_address=args.pub_address,
        sparsenet_weight=args.sparsenet_weight,
        input_size=args.input_size,
        dmax=args.dmax,
        high_pass_noop=args.high_pass_noop,
    )

    if args.offline_test:
        data = np.load(args.offline_test)
        sd = data.get("sparse_depth")
        vm = data.get("valid_mask")
        si = data.get("semantic_id")
        if sd is not None:
            dd = service.run_once_offline(sd, vm, si)
            logger.info(f"Offline test complete. Output min={dd.min():.3f}, max={dd.max():.3f}")
    else:
        service.run()

    service.cleanup()


if __name__ == "__main__":
    main()
