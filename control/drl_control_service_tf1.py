#!/usr/bin/env python3
"""
TF1 DeepRL 控制服务 (orin_drl_tf1)
==================================
职责:
  1. 订阅 ZeroMQ dense_depth_frame (来自 TF/Keras 深度补全)
  2. 将 dense_depth + semantic_id 适配为 DeepRL 观测 (128x128x2 uint8)
  3. 调用原生 stable_baselines.ppo2.PPO2 模型推理动作
  4. 映射动作 → 速度指令 → 发送 MAVSDK/MAVLink

环境: Python 3.6, tensorflow==1.15.0, stable-baselines==2.10.2, gym==0.21.0

创建环境:
  conda create -n orin_drl_tf1 python=3.6 -y
  conda activate orin_drl_tf1
  pip install -r requirements_drl_tf1.txt

ARM/aarch64 (Orin) 注意:
  tensorflow==1.15.0 无官方 aarch64 wheel
  优先: NVIDIA Jetson TF1 wheel
  备选: Docker 容器运行 TF1 环境

动作空间 (10 离散动作, 与 DeepRL 训练一致):
  0: 悬停 (零速度)
  1-8: 八方向水平移动 (45° 间隔)
  9: 垂直下降

WATCHDOG: 超过 watchdog_timeout_s 未收到新帧 → 发送零速度/悬停
"""

import os
import sys
import time
import logging
import argparse
import math
import threading
from typing import Optional, Dict, Any, Tuple

import numpy as np

try:
    import zmq
    HAS_ZMQ = True
except ImportError:
    zmq = None
    HAS_ZMQ = False

# ---- 项目内模块 ----
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control.zmq_protocol import (
    deserialize_dense_depth_frame,
    adapt_to_drl_observation,
    validate_observation,
    HEADER_SIZE,
    OBS_HEIGHT,
    OBS_WIDTH,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("DRLController")


# ================================================================
# 动作映射 (与 DeepRL quadrotor_env.py quadrotor_run() 完全一致)
# ================================================================

def action_to_velocity(action: int, vel_lateral: float = 1.0,
                       vel_vertical: float = 1.0) -> np.ndarray:
    """
    将离散动作映射为 NED 速度指令 [vx, vy, vz] (body frame).

    Action semantics:
      0 → hover (0, 0, 0)
      1-8 → lateral movement in 8 directions
      9 → descend vertically

    返回: np.ndarray [vx, vy, vz] (m/s, NED)
    """
    n_rot = 8
    vel = np.zeros(3, dtype=np.float32)

    if action == 0:
        # 悬停
        pass
    elif 1 <= action <= 8:
        angle = (action - 1) * 2 * math.pi / n_rot
        vel[0] = vel_lateral * math.cos(angle)
        vel[1] = -vel_lateral * math.sin(angle)
        vel[2] = 0.0
    elif action == 9:
        vel[0] = 0.0
        vel[1] = 0.0
        vel[2] = vel_vertical  # positive = downward (NED)
    else:
        raise ValueError(f"Invalid action: {action} (must be 0-9)")

    return vel


# ================================================================
# PPO2 模型加载器 (TF1 / stable_baselines)
# ================================================================

class PPO2Inference:
    """
    加载原生 stable_baselines.ppo2.PPO2 模型进行推理.
    
    输入: (128, 128, 2) uint8 观测 [depth, semantic]
    输出: 离散动作 0-9
    """

    def __init__(self, model_path: str):
        self.model = None
        self._tf_available = False

        try:
            # 抑制 TF1 的冗长日志
            os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
            import tensorflow.compat.v1 as tf
            tf.disable_v2_behavior()
            self.tf = tf

            from stable_baselines.ppo2 import PPO2

            self.model = PPO2.load(model_path)
            self._tf_available = True
            logger.info(f"PPO2 model loaded from {model_path}")
            logger.info(f"  observation_space: {self.model.observation_space}")
            logger.info(f"  action_space: {self.model.action_space}")
        except ImportError as e:
            logger.error(f"Cannot import stable_baselines or tensorflow: {e}")
            raise RuntimeError(
                "PPO2 requires tensorflow==1.x and stable_baselines. "
                "Please activate orin_drl_tf1 conda environment."
            )
        except Exception as e:
            logger.error(f"Failed to load PPO2 model: {e}")
            raise

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """
        推理单帧观测 → 动作索引.

        参数:
          obs: (128, 128, 2) uint8 观测
          deterministic: 是否确定性推理

        返回:
          action: int, 0-9
        """
        if not self._tf_available or self.model is None:
            logger.error("Model not available; returning hover (0)")
            return 0

        # stable_baselines PPO2.predict 期望 obs 是 (n_env, ...) 或单帧
        # 如果 obs 是 (128, 128, 2), 需要 reshape 到 (1, 128, 128, 2) 或 flatten
        obs_space = self.model.observation_space
        obs_shape = obs_space.shape

        if len(obs_shape) == 3:
            # 期望 (H, W, C), 需要加 batch 维度
            obs_input = obs[np.newaxis, ...]
        elif len(obs_shape) == 1:
            # 期望 flatten 后的 (H*W*C,)
            obs_input = obs.flatten()[np.newaxis, ...]
        else:
            obs_input = obs[np.newaxis, ...]

        action, _states = self.model.predict(obs_input, deterministic=deterministic)
        
        if np.ndim(action) == 0:
            return int(action)
        return int(action[0])

    def predict_with_probs(self, obs: np.ndarray) -> Tuple[int, Optional[np.ndarray]]:
        """
        推理并返回动作概率分布 (如果可用).

        返回: (action, probs)
        """
        if not self._tf_available or self.model is None:
            return 0, None

        action, _states = self.model.predict(obs[np.newaxis, ...], deterministic=True)
        action_id = int(action) if np.ndim(action) == 0 else int(action[0])

        # 尝试获取概率 (PPO2 的 action_probability 可能不可用)
        probs = None
        try:
            probs = self.model.action_probability(obs[np.newaxis, ...])
        except Exception:
            pass

        return action_id, probs


# ================================================================
# 飞控接口 (MAVSDK / MAVLink)
# ================================================================

class FlightController:
    """
    飞控抽象接口。

    支持:
      - MAVSDK (Orin 真机)
      - MAVLink (PX4)
      - 空操作 (测试模式, 只打印)
    """

    def __init__(self, mode: str = "noop",
                 mavsdk_address: str = "udp://:14540"):
        self.mode = mode
        self.drone = None

        if mode == "mavsdk":
            try:
                import mavsdk
                from mavsdk import System
                self._mavsdk = mavsdk
                self._mavsdk_system = System(mavsdk_server_address="localhost")
                logger.info(f"MAVSDK initialized (address={mavsdk_address})")
                self.drone = None  # 连接在 connect() 中建立
            except ImportError:
                logger.warning("mavsdk not installed; falling back to noop")
                self.mode = "noop"
        elif mode == "mavlink":
            logger.info("MAVLink mode selected (pass-through)")
        elif mode == "noop":
            logger.info("FlightController in NOOP mode (printing actions only)")

    async def connect(self):
        """异步连接 (仅 MAVSDK 需要)."""
        if self.mode == "mavsdk" and self._mavsdk_system:
            await self._mavsdk_system.connect()
            self.drone = self._mavsdk_system.drone
            logger.info("MAVSDK connected")

    def send_velocity_ned(self, vx: float, vy: float, vz: float,
                          yaw_rate: float = 0.0):
        """
        发送 NED 速度指令。

        参数:
          vx, vy, vz: m/s (NED frame, body)
          yaw_rate: rad/s
        """
        if self.mode == "noop":
            logger.info(f"[FC-NOOP] SendVelNED({vx:.2f}, {vy:.2f}, {vz:.2f}, {yaw_rate:.2f})")
            return

        if self.mode == "mavsdk":
            # MAVSDK 异步调用 (此处简化, 实际需在 event loop 中)
            logger.info(f"[FC-MAVSDK] SendVelNED({vx:.2f}, {vy:.2f}, {vz:.2f}, {yaw_rate:.2f})")
            # TODO: 实际 MAVSDK 调用
            return

        if self.mode == "mavlink":
            # pymavlink 调用
            logger.info(f"[FC-MAVLink] SendVelNED({vx:.2f}, {vy:.2f}, {vz:.2f}, {yaw_rate:.2f})")
            # TODO: 实际 MAVLink 调用
            return


# ================================================================
# DRL 控制服务主类
# ================================================================

class DRLControlService:
    """
    DeepRL 控制服务。
    
    订阅 dense_depth_frame → 适配观测 → PPO2 推理 → 发送控制。
    """

    def __init__(
        self,
        sub_address: str = "tcp://127.0.0.1:5556",
        ppo2_model_path: Optional[str] = None,
        vel_lateral: float = 1.0,
        vel_vertical: float = 1.0,
        dmax: float = 30.0,
        watchdog_timeout_s: float = 0.5,  # 500ms 未收到帧 → 悬停
        fc_mode: str = "noop",
        mavsdk_address: str = "udp://:14540",
        print_stats: bool = True,
    ):
        if not HAS_ZMQ:
            raise ImportError(
                "pyzmq is required for DRLControlService. "
                "Install with: pip install pyzmq"
            )
        # ---- ZeroMQ ----
        self.ctx = zmq.Context()
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.connect(sub_address)
        self.sub.setsockopt(zmq.SUBSCRIBE, b"")
        self.sub.setsockopt(zmq.CONFLATE, 1)  # 只保留最新消息
        logger.info(f"DRLControl: SUB ← {sub_address}")

        # ---- PPO2 模型 ----
        if ppo2_model_path and os.path.exists(ppo2_model_path):
            self.ppo2 = PPO2Inference(ppo2_model_path)
        else:
            logger.error(f"PPO2 model not found at {ppo2_model_path}")
            raise FileNotFoundError(f"Model not found: {ppo2_model_path}")

        # ---- 飞控 ----
        self.fc = FlightController(mode=fc_mode, mavsdk_address=mavsdk_address)

        # ---- 参数 ----
        self.vel_lateral = vel_lateral
        self.vel_vertical = vel_vertical
        self.dmax = dmax
        self.watchdog_timeout_s = watchdog_timeout_s
        self.print_stats = print_stats

        # ---- 状态 ----
        self.last_frame_time = time.time()
        self.last_action = 0
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0
        self.total_inference_ms = 0.0
        self.action_history = [0] * 20  # 用于动作坍缩检测

    def run(self):
        """主循环."""
        logger.info("DRLControlService running...")
        logger.info(f"  vel_lateral={self.vel_lateral}, vel_vertical={self.vel_vertical}")
        logger.info(f"  watchdog_timeout={self.watchdog_timeout_s}s")
        logger.info(f"  fc_mode={self.fc.mode}")

        poller = zmq.Poller()
        poller.register(self.sub, zmq.POLLIN)

        while True:
            try:
                socks = dict(poller.poll(timeout=50))  # 50ms poll
                if self.sub in socks:
                    self._handle_frame()
                else:
                    self._check_watchdog()
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                # 出错时发送零速度
                self._send_hover()

        self.cleanup()

    def _handle_frame(self):
        """处理一帧 dense_depth_frame."""
        t0 = time.time()
        self.last_frame_time = t0

        message = self.sub.recv()
        if len(message) < HEADER_SIZE:
            logger.warning(f"Undersized message: {len(message)} bytes")
            return

        try:
            data = deserialize_dense_depth_frame(
                message[:HEADER_SIZE], message[HEADER_SIZE:]
            )
        except Exception as e:
            logger.error(f"Deserialize failed: {e}")
            return

        dense_depth = data["dense_depth"]
        semantic_id = data["semantic_id"]
        frame_id = data["frame_id"]

        # ---- 适配为 DRL 观测 (128, 128, 2) uint8 ----
        obs = adapt_to_drl_observation(dense_depth, semantic_id, dmax=self.dmax)

        if not validate_observation(obs):
            logger.error(f"Invalid observation shape: {obs.shape}, dtype: {obs.dtype}")
            return

        # ---- PPO2 推理 ----
        inference_start = time.time()
        try:
            action = self.ppo2.predict(obs, deterministic=True)
        except Exception as e:
            logger.error(f"PPO2 inference failed: {e}")
            action = 0  # fallback to hover
        inference_ms = (time.time() - inference_start) * 1000

        # ---- 动作映射 → 速度 ----
        vel = action_to_velocity(action, self.vel_lateral, self.vel_vertical)

        # ---- 发送控制 ----
        self.fc.send_velocity_ned(vel[0], vel[1], vel[2])

        # ---- 统计 ----
        self.last_action = action
        self.action_history.append(action)
        self.action_history.pop(0)
        self.frame_count += 1
        self.total_inference_ms += inference_ms

        now = time.time()
        if now - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (now - self.last_fps_time)
            avg_inf_ms = self.total_inference_ms / max(self.frame_count, 1)
            total_latency_ms = (t0 - data["timestamp_ns"] / 1e9) * 1000

            if self.print_stats:
                logger.info(
                    f"[DRL] FPS={self.fps:.1f} | action={action} | "
                    f"vel=({vel[0]:.2f},{vel[1]:.2f},{vel[2]:.2f}) | "
                    f"inf={inference_ms:.1f}ms avg={avg_inf_ms:.1f}ms | "
                    f"e2e_lat={total_latency_ms:.0f}ms | "
                    f"depth=[{dense_depth.min():.1f},{dense_depth.max():.1f}]m"
                )

            # 动作坍缩检测
            unique_actions = set(self.action_history)
            if len(unique_actions) == 1:
                logger.warning(
                    f"[DRL] ACTION COLLAPSE: all last 20 actions = {self.action_history[0]}"
                )

            self.frame_count = 0
            self.total_inference_ms = 0.0
            self.last_fps_time = now

    def _check_watchdog(self):
        """检查是否超时, 超时则发送悬停."""
        elapsed = time.time() - self.last_frame_time
        if elapsed > self.watchdog_timeout_s:
            logger.warning(
                f"[DRL] WATCHDOG: {elapsed:.3f}s since last frame. Sending HOVER."
            )
            self._send_hover()
            self.last_frame_time = time.time()  # 避免重复触发

    def _send_hover(self):
        """发送零速度指令."""
        self.fc.send_velocity_ned(0.0, 0.0, 0.0)
        self.last_action = 0

    def run_once_offline(self, dense_depth: np.ndarray,
                         semantic_id: np.ndarray) -> int:
        """离线单帧测试: 输入深度+语义, 返回动作."""
        obs = adapt_to_drl_observation(dense_depth, semantic_id, dmax=self.dmax)
        action = self.ppo2.predict(obs, deterministic=True)
        vel = action_to_velocity(action, self.vel_lateral, self.vel_vertical)
        logger.info(
            f"Offline test: action={action}, "
            f"vel=({vel[0]:.2f},{vel[1]:.2f},{vel[2]:.2f}), "
            f"depth_range=[{dense_depth.min():.2f},{dense_depth.max():.2f}]m"
        )
        return action

    def cleanup(self):
        """安全关闭."""
        logger.info("Shutting down DRLControlService...")
        self._send_hover()
        time.sleep(0.1)
        self.sub.close()
        self.ctx.term()


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="DRL Control Service (TF1/stable_baselines)")
    parser.add_argument("--sub-address", default="tcp://127.0.0.1:5556",
                        help="SUB address (from depth completion)")
    parser.add_argument("--ppo2-model", required=True,
                        help="Path to PPO2 last_step_model.zip")
    parser.add_argument("--vel-lateral", type=float, default=1.0,
                        help="Lateral velocity (m/s)")
    parser.add_argument("--vel-vertical", type=float, default=1.0,
                        help="Vertical velocity (m/s)")
    parser.add_argument("--dmax", type=float, default=30.0,
                        help="Max depth (m)")
    parser.add_argument("--watchdog-ms", type=float, default=500.0,
                        help="Watchdog timeout (ms)")
    parser.add_argument("--fc-mode", default="noop",
                        choices=["noop", "mavsdk", "mavlink"],
                        help="Flight controller mode")
    parser.add_argument("--mavsdk-address", default="udp://:14540",
                        help="MAVSDK address")
    parser.add_argument("--offline-test", default=None,
                        help="Path to .npz with dense_depth/semantic_id for offline test")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-frame stats")
    args = parser.parse_args()

    service = DRLControlService(
        sub_address=args.sub_address,
        ppo2_model_path=args.ppo2_model,
        vel_lateral=args.vel_lateral,
        vel_vertical=args.vel_vertical,
        dmax=args.dmax,
        watchdog_timeout_s=args.watchdog_ms / 1000.0,
        fc_mode=args.fc_mode,
        mavsdk_address=args.mavsdk_address,
        print_stats=not args.quiet,
    )

    if args.offline_test:
        data = np.load(args.offline_test)
        dd = data.get("dense_depth")
        si = data.get("semantic_id")
        if dd is not None and si is not None:
            action = service.run_once_offline(dd, si)
            logger.info(f"Offline test action: {action}")
        else:
            logger.error("Offline test .npz must contain 'dense_depth' and 'semantic_id'")
    else:
        service.run()

    service.cleanup()


if __name__ == "__main__":
    main()
