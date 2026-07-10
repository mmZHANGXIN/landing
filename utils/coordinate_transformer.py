"""
坐标转换工具 - 从 FaultyYawLanding/utils/geometric_utils.py 移植优化
适配 Orin 部署, 移除 RflySim 依赖, 使用标准 NumPy
"""

import numpy as np
import math


def euler_to_quaternion(r: float, p: float, y: float):
    """欧拉角 → 四元数 [w, x, y, z]"""
    cy, sy = math.cos(y * 0.5), math.sin(y * 0.5)
    cp, sp = math.cos(p * 0.5), math.sin(p * 0.5)
    cr, sr = math.cos(r * 0.5), math.sin(r * 0.5)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


def get_rotation_matrix(r: float, p: float, y: float) -> np.ndarray:
    """欧拉角 → 旋转矩阵 R = Rz(y) @ Ry(p) @ Rx(r)"""
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


class CoordinateTransformer:
    """
    针孔相机模型坐标转换器
    - 像素 ↔ 机体坐标投影
    - 世界 ↔ 像素投影
    """

    def __init__(self, fov: float, width: int, height: int, camera_extrinsics: np.ndarray):
        self.w = width
        self.h = height
        # 内参: 焦距 f = W / (2 * tan(FOV/2))
        f = width / (2.0 * np.tan(np.deg2rad(fov) / 2.0))
        self.K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]])
        self.T_cb = camera_extrinsics  # Camera → Body 变换 (4x4)

    def pixel_to_body_ground(self, u: float, v: float, drone_height: float) -> np.ndarray:
        """像素坐标 → 机体地面投影点 (假设平坦地面)"""
        uv = np.array([u, v, 1.0])
        norm_uv = np.linalg.inv(self.K) @ uv
        scale = drone_height / max(norm_uv[2], 1e-6)
        return norm_uv * scale

    def world_to_pixel(self, xyz_world: np.ndarray, drone_pose_3dof: np.ndarray) -> tuple:
        """世界坐标 → 像素坐标 (u, v)"""
        x, y, z, r, p, yaw = drone_pose_3dof
        R = get_rotation_matrix(r, p, yaw)
        T_wb = np.eye(4)
        T_wb[:3, :3] = R
        T_wb[:3, 3] = [x, y, z]

        T_wc = T_wb @ self.T_cb
        T_cw = np.linalg.inv(T_wc)

        p_world = np.append(xyz_world, 1.0)
        p_cam = T_cw @ p_world

        if p_cam[2] <= 1e-6:
            return -1, -1

        uv_homo = self.K @ (p_cam[:3] / p_cam[2])
        return int(uv_homo[0]), int(uv_homo[1])

    def body_to_pixel(self, xyz_body: np.ndarray) -> tuple:
        """机体坐标 → 像素坐标"""
        p_cam = np.linalg.inv(self.T_cb) @ np.append(xyz_body, 1.0)
        if p_cam[2] <= 1e-6:
            return -1, -1
        uv_homo = self.K @ (p_cam[:3] / p_cam[2])
        return int(uv_homo[0]), int(uv_homo[1])
