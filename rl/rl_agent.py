"""
SB3 PPO 深度强化学习推理代理
从原 DeepRL 项目迁移，使用 stable-baselines3。

输入: (128, 128, 2) 观测张量 [depth, semantics]
输出: 离散动作 0-9
"""

import numpy as np
import torch
import cv2
import logging

logger = logging.getLogger("RLAgent")

SEMANTIC_GRAY = {
    -1: 0.0,  # unknown
    0: 10.0,
    1: 30.0,  # terrain / safe
    2: 60.0,
    3: 70.0,
    4: 20.0,
    5: 40.0,
    6: 80.0,
    7: 90.0,
    8: 50.0,
    9: 250.0,  # others / danger
}


class RLAgent:
    """
    轻量 RL 推理代理。

    动作空间 (10 离散动作):
      0: 悬停
      1-8: 8 方向水平移动 (间隔 45°)
      9: 下降
    """

    def __init__(self, model_path: str, img_size: tuple = (128, 128),
                 vel_lateral: float = 1.0, vel_vertical: float = 10.0,
                 dmax: float = 30.0, depth_norm_mode: str = "unit",
                 semantic_norm_mode: str = "unit", require_gpu: bool = False):
        self.img_w, self.img_h = img_size
        self.vel_lateral = vel_lateral
        self.vel_vertical = vel_vertical
        self.dmax = float(dmax)
        self.depth_norm_mode = depth_norm_mode
        self.semantic_norm_mode = semantic_norm_mode
        self.require_gpu = bool(require_gpu)
        self.obs_layout = "hwc"
        self.model_obs_shape = None
        cuda_ok = torch.cuda.is_available()
        if self.require_gpu and not cuda_ok:
            raise RuntimeError(
                "[RLAgent] CUDA NOT available. "
                "SB3/PyTorch policy CPU fallback is denied for flight."
            )
        device = 'cuda' if cuda_ok else 'cpu'
        self._device = device

        # 加载 SB3 模型
        try:
            from stable_baselines3 import PPO
            logger.info(
                f"[RLAgent] Loading model from {model_path} on {device} "
                f"(require_gpu={self.require_gpu})..."
            )
            self.model = PPO.load(model_path, device=device)
            self.model_obs_shape = tuple(self.model.observation_space.shape)
            if len(self.model_obs_shape) == 3 and self.model_obs_shape[0] == 2:
                self.obs_layout = "chw"
            elif len(self.model_obs_shape) == 3 and self.model_obs_shape[-1] == 2:
                self.obs_layout = "hwc"
            else:
                logger.warning("[RLAgent] Unexpected observation space shape: %s",
                               self.model_obs_shape)
            logger.info("[RLAgent] Model loaded.")
            logger.info("[RLAgent] obs_shape=%s layout=%s depth_mode=%s sem_mode=%s",
                        self.model_obs_shape, self.obs_layout,
                        self.depth_norm_mode, self.semantic_norm_mode)
        except Exception as e:
            if self.require_gpu:
                raise RuntimeError(
                    f"[RLAgent] FAILED to load required GPU policy from {model_path}."
                ) from e
            logger.warning(f"[RLAgent] Failed to load model: {e}. Using dummy policy.")
            self.model = None
            self._device = 'cpu'

    def _preprocess(self, depth_map: np.ndarray, sem_map: np.ndarray) -> np.ndarray:
        """
        拼接深度和语义为观测张量 (H, W, 2)。
        """
        # 确保尺寸一致
        if depth_map.shape != (self.img_h, self.img_w):
            depth_map = cv2.resize(depth_map, (self.img_w, self.img_h),
                                   interpolation=cv2.INTER_LINEAR)
        if sem_map.shape != (self.img_h, self.img_w):
            sem_map = cv2.resize(sem_map, (self.img_w, self.img_h),
                                 interpolation=cv2.INTER_NEAREST)

        depth_norm, sem_norm = self._encode_maps(depth_map, sem_map)

        if self.obs_layout == "chw":
            obs = np.stack([depth_norm, sem_norm], axis=0)  # (2, H, W)
        else:
            obs = np.stack([depth_norm, sem_norm], axis=-1)  # (H, W, 2)
        return obs

    def _encode_maps(self, depth_map: np.ndarray, sem_map: np.ndarray):
        """Encode depth/semantic maps in the convention expected by the policy."""
        return self._encode_depth(depth_map), self._encode_semantic(sem_map)

    def _encode_depth(self, depth_map: np.ndarray) -> np.ndarray:
        depth = np.nan_to_num(depth_map, nan=self.dmax, posinf=self.dmax, neginf=0.0)
        if self.depth_norm_mode == "unit":
            return np.clip(depth / self.dmax, 0.0, 1.0).astype(np.float32)
        if self.depth_norm_mode == "inverse_unit":
            return np.clip(1.0 - depth / self.dmax, 0.0, 1.0).astype(np.float32)
        if self.depth_norm_mode == "meters":
            return np.clip(depth, 0.0, self.dmax).astype(np.float32)
        if self.depth_norm_mode == "meters_div255":
            # Original SB2 policy used scale=True on a Box(0, 255) image space.
            # When raw depth was stored in meters, the policy saw depth / 255.
            return np.clip(depth / 255.0, 0.0, 1.0).astype(np.float32)
        raise ValueError(f"Unsupported depth_norm_mode: {self.depth_norm_mode}")

    def _encode_semantic(self, sem_map: np.ndarray) -> np.ndarray:
        sem = sem_map.astype(np.float32)
        if self.semantic_norm_mode == "unit":
            return np.clip(sem / 9.0, 0.0, 1.0).astype(np.float32)
        if self.semantic_norm_mode == "raw":
            return sem.astype(np.float32)
        if self.semantic_norm_mode in ("gray_unit", "gray_raw"):
            if np.nanmax(sem) > 9.0:
                gray = np.clip(sem, 0.0, 255.0).astype(np.float32)
            else:
                sem_i = sem_map.astype(np.int16)
                gray = np.zeros_like(sem, dtype=np.float32)
                for class_id, gray_value in SEMANTIC_GRAY.items():
                    gray[sem_i == class_id] = gray_value
            if self.semantic_norm_mode == "gray_unit":
                gray = gray / 255.0
            return np.clip(gray, 0.0, 255.0).astype(np.float32)
        raise ValueError(f"Unsupported semantic_norm_mode: {self.semantic_norm_mode}")

    def predict(self, depth_map: np.ndarray, sem_map: np.ndarray) -> int:
        """
        执行推理，返回动作索引 0-9。
        """
        obs = self._preprocess(depth_map, sem_map)

        if self.model is not None:
            # SB3 期望 (H, W, C) 或 (n_env, H, W, C)
            with torch.no_grad():
                action, _ = self.model.predict(obs, deterministic=True)
            return int(action) if np.ndim(action) == 0 else int(action[0])
        else:
            # 无模型时返回悬停
            return 0

    def predict_with_info(self, depth_map: np.ndarray, sem_map: np.ndarray):
        """
        执行推理并返回诊断信息。

        返回:
          action: int
          info: dict, 包含 obs 统计和可用时的动作概率
        """
        obs = self._preprocess(depth_map, sem_map)
        info = self.get_observation_stats(depth_map, sem_map)

        if self.model is None:
            info["action_probs"] = None
            info["confidence"] = None
            return 0, info

        with torch.no_grad():
            action, _ = self.model.predict(obs, deterministic=True)
        action_id = int(action) if np.ndim(action) == 0 else int(action[0])
        probs = self._try_action_probs(obs)
        info["action_probs"] = probs
        info["confidence"] = None if probs is None else float(np.max(probs))
        return action_id, info

    def get_observation_stats(self, depth_map: np.ndarray, sem_map: np.ndarray) -> dict:
        """返回 DRL 输入统计, 用于定位动作锁定/观测退化问题。"""
        if depth_map.shape != (self.img_h, self.img_w):
            depth_map = cv2.resize(depth_map, (self.img_w, self.img_h),
                                   interpolation=cv2.INTER_LINEAR)
        if sem_map.shape != (self.img_h, self.img_w):
            sem_map = cv2.resize(sem_map, (self.img_w, self.img_h),
                                 interpolation=cv2.INTER_NEAREST)
        depth, sem = self._encode_maps(depth_map, sem_map)
        return {
            "depth_norm_mode": self.depth_norm_mode,
            "semantic_norm_mode": self.semantic_norm_mode,
            "obs_layout": self.obs_layout,
            "model_obs_shape": self.model_obs_shape,
            "depth_min": float(np.min(depth_map)),
            "depth_mean": float(np.mean(depth_map)),
            "depth_max": float(np.max(depth_map)),
            "depth_norm_min": float(np.min(depth)),
            "depth_norm_mean": float(np.mean(depth)),
            "depth_norm_max": float(np.max(depth)),
            "sem_norm_min": float(np.min(sem)),
            "sem_norm_mean": float(np.mean(sem)),
            "sem_norm_max": float(np.max(sem)),
        }

    def _try_action_probs(self, obs: np.ndarray):
        """Best-effort 获取 PPO categorical action 概率。失败时返回 None。"""
        try:
            obs_tensor, _ = self.model.policy.obs_to_tensor(obs)
            dist = self.model.policy.get_distribution(obs_tensor)
            probs_t = getattr(dist.distribution, "probs", None)
            if probs_t is None:
                return None
            probs = probs_t.detach().cpu().numpy()
            return probs[0].astype(float).tolist() if probs.ndim > 1 else probs.astype(float).tolist()
        except Exception as e:
            logger.debug(f"[RLAgent] Action probability unavailable: {e}")
            return None

    def map_action_to_velocity(self, action: int) -> np.ndarray:
        """
        离散动作 → 世界系 NED 速度指令 [vx, vy, vz]。

        动作语义 (与原 DeepRL quadrotor_env.py 一致):
          0: 悬停
          1-8: 8方向水平移动, vel_y = -sin(angle)
          9: 垂直下降

        MAVSDK set_velocity_ned 期望 NED (世界系), PX4 内部做姿态转换。
        真机主管线使用 control.ActionDecomposer 做机体系 yaw 补偿。
        """
        n_rot = 8
        vel = np.zeros(3, dtype=np.float32)

        if action == 0:
            pass  # 悬停
        elif 1 <= action <= 8:
            angle = (action - 1) * 2.0 * np.pi / n_rot
            vel[0] = self.vel_lateral * np.cos(angle)   # North
            vel[1] = -self.vel_lateral * np.sin(angle)  # East (NED: East+ = -sin)
            vel[2] = 0.0
        elif action == 9:
            vel[2] = self.vel_vertical  # Down+ = 下降
        else:
            raise ValueError(f"Invalid action: {action}")

        return vel

    def apply_yaw_compensation(self, vel_ned: np.ndarray, yaw: float) -> np.ndarray:
        """
        可选: 将世界系 NED 速度旋转到机体系 (适配 RflySim 风格的训练策略)。

        如果训练时策略输出的是"世界系意图"但仿真环境用的是机体系指令,
        则需要此补偿。对于 MAVSDK (原生 NED), 通常不需要。

        v_body = R_z(yaw)^T @ v_world
        """
        import math
        c = math.cos(yaw)
        s = math.sin(yaw)
        vx_b = c * vel_ned[0] + s * vel_ned[1]
        vy_b = -s * vel_ned[0] + c * vel_ned[1]
        return np.array([vx_b, vy_b, vel_ned[2]], dtype=np.float32)
