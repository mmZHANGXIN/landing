#!/usr/bin/env python3
"""
ONNX DRL 推理服务 (orin_drl_onnx)
=================================
Orin aarch64 原生 DRL 推理 — 零 TensorFlow 依赖！

工作流:
  1. 加载从 x86 TF1 PPO2 导出的 ONNX 模型
  2. 订阅 ZeroMQ dense_depth_frame (来自深度补全)
  3. dense_depth + semantic → 适配 128x128x2 obs → ONNX推理 → 动作映射

依赖 (全有 aarch64 wheel):
  numpy, opencv-python, pyzmq, onnxruntime

创建环境:
  conda create -n drl_onnx python=3.8 -y
  conda activate drl_onnx
  pip install numpy opencv-python pyzmq onnxruntime pyyaml

用法:
  python control/drl_control_service_onnx.py \
      --onnx-model weights/ppo2_policy.onnx \
      --sub-address tcp://127.0.0.1:5556 \
      --fc-mode noop
"""

import os
import sys
import time
import math
import json
import logging
import argparse
from typing import Optional, Dict, Tuple

import numpy as np

# ---- 延迟导入 (环境可能不全) ----
try:
    import zmq
    HAS_ZMQ = True
except ImportError:
    zmq = None
    HAS_ZMQ = False

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    ort = None
    HAS_ONNX = False

# ---- 项目模块 ----
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control.zmq_protocol import (
    deserialize_dense_depth_frame,
    adapt_to_drl_observation,
    validate_observation,
    HEADER_SIZE,
    OBS_HEIGHT,
    OBS_WIDTH,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger("DRL_ONNX")


# ================================================================
# 动作映射 (与 DeepRL quadrotor_env.quadrotor_run 100% 一致)
# ================================================================

ACTION_MAP = {
    0: "HOVER",
    1: "MOVE_N",
    2: "MOVE_NE",
    3: "MOVE_E",
    4: "MOVE_SE",
    5: "MOVE_S",
    6: "MOVE_SW",
    7: "MOVE_W",
    8: "MOVE_NW",
    9: "DESCEND",
}


def action_to_velocity(action: int, vel_lateral: float = 1.0,
                       vel_vertical: float = 1.0) -> np.ndarray:
    """离散动作 → NED 速度 [vx, vy, vz]."""
    n_rot = 8
    vel = np.zeros(3, dtype=np.float32)
    if action == 0:
        pass
    elif 1 <= action <= 8:
        angle = (action - 1) * 2 * math.pi / n_rot
        vel[0] = vel_lateral * math.cos(angle)
        vel[1] = -vel_lateral * math.sin(angle)
        vel[2] = 0.0
    elif action == 9:
        vel[0] = 0.0; vel[1] = 0.0; vel[2] = vel_vertical
    else:
        raise ValueError(f"Invalid action {action}, valid: 0–9")
    return vel


# ================================================================
# ONNX 推理器
# ================================================================

class ONNXPolicy:
    """
    从 TF1 PPO2 导出的 ONNX 策略网络推理器。

    输入:  (N, H, W, C) float32, range [0, 1]  (obs/255)
    输出:  (N, 10) float32 action logits
    """

    def __init__(self, onnx_path: str, meta_path: Optional[str] = None):
        if not HAS_ONNX:
            raise ImportError("pip install onnxruntime")

        # --- 加载 ONNX ---
        logger.info(f"Loading ONNX: {onnx_path}")
        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, providers=providers)

        self.input_info = self.session.get_inputs()[0]
        self.output_info = self.session.get_outputs()[0]
        self.input_name = self.input_info.name
        self.output_name = self.output_info.name

        # 推断输入形状
        in_shape = self.input_info.shape
        logger.info(f"  ONNX input:  {self.input_name} shape={in_shape}")
        logger.info(f"  ONNX output: {self.output_name} shape={self.output_info.shape}")

        # 确定观测布局
        if len(in_shape) == 2:
            self.layout = "flat"            # (1, 32768)
        elif len(in_shape) == 4:
            if in_shape[1] in (2, 3):
                self.layout = "chw"         # (1, 2, 128, 128)
            else:
                self.layout = "hwc"         # (1, 128, 128, 2)
        else:
            self.layout = "flat"

        logger.info(f"  Inferred layout: {self.layout}")

        # --- 元数据 ---
        self.meta = {}
        if meta_path and os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                self.meta = json.load(f)
            logger.info(f"  action_space_n: {self.meta.get('action_space_n')}")

        # --- 预热 ---
        self._warmup()

    def _warmup(self):
        obs = np.zeros((1, OBS_HEIGHT, OBS_WIDTH, 2), dtype=np.float32)
        try:
            self._forward(obs)
            logger.info("  ONNX warmup OK")
        except Exception as e:
            logger.warning(f"  ONNX warmup failed: {e}")

    def _forward(self, obs_norm: np.ndarray) -> np.ndarray:
        """obs_norm: (N, H, W, C) float32, range [0, 1]"""
        if self.layout == "flat":
            inp = obs_norm.reshape(obs_norm.shape[0], -1).astype(np.float32)
        elif self.layout == "chw":
            inp = np.transpose(obs_norm, (0, 3, 1, 2)).astype(np.float32)
        else:
            inp = obs_norm.astype(np.float32)

        outputs = self.session.run([self.output_name], {self.input_name: inp})
        return outputs[0]

    def predict(self, obs: np.ndarray) -> int:
        """
        单帧推理 → 动作索引 0-9。

        obs: (128, 128, 2) uint8
        """
        # 归一化: Box(0,255) → [0,1]
        obs_f = obs.astype(np.float32) / 255.0
        obs_batch = obs_f[np.newaxis, ...]  # (1, 128, 128, 2)

        logits = self._forward(obs_batch)   # (1, 10)
        action = int(np.argmax(logits[0]))
        return action

    def predict_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        """批量推理 → 动作数组."""
        obs_f = obs_batch.astype(np.float32) / 255.0
        logits = self._forward(obs_f)
        return np.argmax(logits, axis=1)

    def predict_with_probs(self, obs: np.ndarray) -> Tuple[int, np.ndarray]:
        """推理 + 返回动作概率分布."""
        obs_f = obs.astype(np.float32) / 255.0
        obs_batch = obs_f[np.newaxis, ...]
        logits = self._forward(obs_batch)
        probs = np.exp(logits[0]) / np.sum(np.exp(logits[0]))  # softmax
        action = int(np.argmax(logits[0]))
        return action, probs


# ================================================================
# 飞控接口
# ================================================================

class FlightController:
    def __init__(self, mode: str = "noop"):
        self.mode = mode
        logger.info(f"FlightController: mode={mode}")

    def send_velocity_ned(self, vx: float, vy: float, vz: float,
                          yaw_rate: float = 0.0):
        if self.mode == "noop":
            logger.info(f"[FC] SendVelNED({vx:.2f}, {vy:.2f}, {vz:.2f})")
        elif self.mode == "mavsdk":
            logger.info(f"[FC-MAVSDK] ({vx:.2f}, {vy:.2f}, {vz:.2f})")
            # TODO: 实际 MAVSDK 调用
        elif self.mode == "mavlink":
            logger.info(f"[FC-MAVLink] ({vx:.2f}, {vy:.2f}, {vz:.2f})")
            # TODO: 实际 MAVLink 调用


# ================================================================
# ONNX DRL 控制服务
# ================================================================

class DRLControlONNXService:
    """ONNX 版 DRL 控制服务 (Orin aarch64 原生)."""

    def __init__(
        self,
        sub_address: str = "tcp://127.0.0.1:5556",
        onnx_model: str = "weights/ppo2_policy.onnx",
        onnx_meta: Optional[str] = None,
        vel_lateral: float = 1.0,
        vel_vertical: float = 1.0,
        dmax: float = 30.0,
        watchdog_timeout_s: float = 0.5,
        fc_mode: str = "noop",
        print_stats: bool = True,
    ):
        if not HAS_ZMQ:
            raise ImportError("pyzmq required: pip install pyzmq")
        if not HAS_ONNX:
            raise ImportError("onnxruntime required: pip install onnxruntime")

        # ---- ZeroMQ SUB ----
        self.ctx = zmq.Context()
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.connect(sub_address)
        self.sub.setsockopt(zmq.SUBSCRIBE, b"")
        self.sub.setsockopt(zmq.CONFLATE, 1)  # 只保留最新帧
        logger.info(f"SUB ← {sub_address}")

        # ---- ONNX 策略 ----
        self.policy = ONNXPolicy(onnx_model, onnx_meta)

        # ---- 飞控 ----
        self.fc = FlightController(mode=fc_mode)

        # ---- 参数 ----
        self.vel_lateral = vel_lateral
        self.vel_vertical = vel_vertical
        self.dmax = dmax
        self.watchdog_timeout_s = watchdog_timeout_s
        self.print_stats = print_stats

        # ---- 运行时状态 ----
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.total_inf_ms = 0.0
        self.action_history = [0] * 20

    # ================================================================
    # 主循环
    # ================================================================

    def run(self):
        logger.info("ONNX DRL Control Service running")
        logger.info(f"  vel_lateral={self.vel_lateral}  vel_vertical={self.vel_vertical}")
        logger.info(f"  watchdog={self.watchdog_timeout_s}s  fc_mode={self.fc.mode}")

        poller = zmq.Poller()
        poller.register(self.sub, zmq.POLLIN)

        try:
            while True:
                socks = dict(poller.poll(timeout=50))
                if self.sub in socks:
                    self._handle_frame()
                else:
                    self._check_watchdog()
        except KeyboardInterrupt:
            logger.info("Interrupted")
        finally:
            self.cleanup()

    def _handle_frame(self):
        t0 = time.time()
        self.last_frame_time = t0

        # 接收
        msg = self.sub.recv()
        if len(msg) < HEADER_SIZE:
            return
        try:
            data = deserialize_dense_depth_frame(msg[:HEADER_SIZE], msg[HEADER_SIZE:])
        except Exception as e:
            logger.error(f"Deserialize: {e}")
            return

        # 适配观测
        obs = adapt_to_drl_observation(
            data["dense_depth"], data["semantic_id"], dmax=self.dmax
        )
        if not validate_observation(obs):
            logger.error(f"Bad obs: {obs.shape} {obs.dtype}")
            return

        # ONNX 推理
        inf_t0 = time.time()
        action = self.policy.predict(obs)
        inf_ms = (time.time() - inf_t0) * 1000

        # 动作 → 速度
        vel = action_to_velocity(action, self.vel_lateral, self.vel_vertical)
        self.fc.send_velocity_ned(vel[0], vel[1], vel[2])

        # ---- 统计 ----
        self.action_history.append(action)
        self.action_history.pop(0)
        self.frame_count += 1
        self.total_inf_ms += inf_ms

        now = time.time()
        if now - self.last_fps_time >= 1.0:
            fps = self.frame_count / (now - self.last_fps_time)
            avg_inf = self.total_inf_ms / max(self.frame_count, 1)
            e2e_lat = (t0 - data["timestamp_ns"] / 1e9) * 1000

            if self.print_stats:
                logger.info(
                    f"[DRL-ONNX] FPS={fps:.1f} action={action} "
                    f"({ACTION_MAP.get(action, '?')}) "
                    f"vel=({vel[0]:.2f},{vel[1]:.2f},{vel[2]:.2f}) "
                    f"inf={inf_ms:.1f}ms avg={avg_inf:.1f}ms e2e={e2e_lat:.0f}ms"
                )

            # 动作坍缩告警
            if len(set(self.action_history)) == 1:
                logger.warning(
                    f"ACTION COLLAPSE: 20 consecutive action={self.action_history[0]}"
                )

            self.frame_count = 0
            self.total_inf_ms = 0.0
            self.last_fps_time = now

    def _check_watchdog(self):
        elapsed = time.time() - self.last_frame_time
        if elapsed > self.watchdog_timeout_s:
            logger.warning(f"WATCHDOG: {elapsed:.1f}s no frame → HOVER")
            self.fc.send_velocity_ned(0.0, 0.0, 0.0)
            self.last_frame_time = time.time()

    def cleanup(self):
        logger.info("Shutting down...")
        self.fc.send_velocity_ned(0.0, 0.0, 0.0)
        time.sleep(0.1)
        self.sub.close()
        self.ctx.term()


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="ONNX DRL Control Service")
    parser.add_argument("--onnx-model", required=True, help="ONNX model path")
    parser.add_argument("--onnx-meta", default=None, help="Meta JSON path")
    parser.add_argument("--sub-address", default="tcp://127.0.0.1:5556")
    parser.add_argument("--vel-lateral", type=float, default=1.0)
    parser.add_argument("--vel-vertical", type=float, default=1.0)
    parser.add_argument("--dmax", type=float, default=30.0)
    parser.add_argument("--watchdog-ms", type=float, default=500.0)
    parser.add_argument("--fc-mode", default="noop",
                        choices=["noop", "mavsdk", "mavlink"])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    service = DRLControlONNXService(
        sub_address=args.sub_address,
        onnx_model=args.onnx_model,
        onnx_meta=args.onnx_meta,
        vel_lateral=args.vel_lateral,
        vel_vertical=args.vel_vertical,
        dmax=args.dmax,
        watchdog_timeout_s=args.watchdog_ms / 1000.0,
        fc_mode=args.fc_mode,
        print_stats=not args.quiet,
    )
    service.run()
    service.cleanup()


if __name__ == "__main__":
    main()
