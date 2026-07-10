"""
DRL 动作解算: 离散动作 ID → 机体系速度 → NED 速度 + 偏航转速
=============================================================
满足需求:
  - DRL 横移动作解释为机体系速度, 用 FastLIO yaw 转为 NED
  - 默认横向符号与原 DeepRL quadrotor_env.py 保持一致:
    vel_des[1] = -vel_lateral * sin(angle)
  - 下降保持 NED down 正方向
  - yaw=0/90/180 单元测试速度方向正确
  - 日志打印转换前后速度

动作集 (10 离散, 默认 DeepRL 符号 action_lateral_sign=-1):
  0: hover   (v=0)
  1: N       (机头方向)
  2: NW
  3: W
  4: SW
  5: S
  6: SE
  7: E
  8: NE
  9: descend (仅下降, 无水平移动)
"""

from __future__ import annotations

import math
from typing import Tuple

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


# 8 方向角度 (弧度): 0=forward(N), CCW
_DIRECTION_ANGLES = (
    math.radians(0),    #  0: N
    math.radians(45),   #  1: NE
    math.radians(90),   #  2: E
    math.radians(135),  #  3: SE
    math.radians(180),  #  4: S
    math.radians(225),  #  5: SW
    math.radians(270),  #  6: W
    math.radians(315),  #  7: NW
)


def _vec(values):
    if np is not None:
        return np.array(values, dtype=np.float32)
    return [float(v) for v in values]


def _zeros():
    if np is not None:
        return np.zeros(3, dtype=np.float32)
    return [0.0, 0.0, 0.0]


def _allclose(a, b, atol=1e-6):
    if np is not None:
        return np.allclose(a, b, atol=atol)
    return all(abs(float(x) - float(y)) <= atol for x, y in zip(a, b))

_ACTION_NAMES_BY_SIGN = {
    -1: ["HOVER", "N", "NW", "W", "SW", "S", "SE", "E", "NE", "DESCEND"],
    1: ["HOVER", "N", "NE", "E", "SE", "S", "SW", "W", "NW", "DESCEND"],
}


class ActionDecomposer:
    """
    动作解算器: DRL 动作 → NED 速度指令 + yaw_rate
    """

    def __init__(self, cfg: dict):
        self.v_lat = cfg.get("vel_lateral", 1.0)       # 水平速度 (m/s)
        self.v_vert = cfg.get("vel_vertical", 1.0)      # 垂直速度 (m/s)
        self.yaw_rate = cfg.get("yaw_rate_rad_s", 0.0)  # 偏航转速 (rad/s)
        self.action_frame = cfg.get("action_frame", "body")  # "body" | "ned"
        if self.action_frame not in ("body", "ned"):
            raise ValueError(f"Unsupported action_frame: {self.action_frame}")
        self.action_lateral_sign = int(cfg.get("action_lateral_sign", -1))
        if self.action_lateral_sign not in (-1, 1):
            raise ValueError(
                f"Unsupported action_lateral_sign: {self.action_lateral_sign}. "
                "Use -1 for original DeepRL or +1 for body-right-positive mirror."
            )
        self.action_names = _ACTION_NAMES_BY_SIGN[self.action_lateral_sign]

    def action_id_to_name(self, action_id: int) -> str:
        return self.action_names[action_id] if 0 <= action_id < len(self.action_names) else "?"

    def decompose(self, action_id: int, yaw_rad: float
                  ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        将 DRL 动作分解为 NED 速度指令。

        Args:
            action_id: 离散动作 ID (0-9)
            yaw_rad:   当前无人机 yaw 角 (弧度, NED: 0=N, CCW+)

        Returns:
            v_body:    (3,) 机体系速度 [forward, right, down]
            v_ned:     (3,) NED 速度 [north, east, down]
            yaw_rate:  yaw 转速 (rad/s)
        """
        if action_id == 0:  # hover
            v_body = _zeros()
            yr = self.yaw_rate
        elif action_id == 9:  # descend
            v_body = _vec([0.0, 0.0, self.v_vert])
            yr = self.yaw_rate
        else:  # 横向移动 (1-8)
            angle = _DIRECTION_ANGLES[action_id - 1]
            v_body = _vec([
                math.cos(angle) * self.v_lat,
                self.action_lateral_sign * math.sin(angle) * self.v_lat,
                0.0,  # 下降由单独动作控制
            ])
            yr = self.yaw_rate

        if self.action_frame == "body":
            # 机体系 → NED 旋转，适配偏航失控下的机体动作意图。
            v_ned = self._body_to_ned(v_body, yaw_rad)
        else:
            # NED模式: 动作直接解释为世界系方向，v_body仅用于日志诊断。
            v_ned = v_body.copy()
            v_body = self._ned_to_body(v_ned, yaw_rad)

        return v_body, v_ned, yr

    def _body_to_ned(self, v_body: np.ndarray, yaw_rad: float) -> np.ndarray:
        """
        机体系速度 → NED 速度

        机体系: x=forward, y=right, z=down
        NED:     x=north,   y=east,  z=down

        旋转: 绕 z 轴旋转 yaw (NED: CCW 为正)
          v_ned_x =  cos(yaw)*v_body_x - sin(yaw)*v_body_y
          v_ned_y =  sin(yaw)*v_body_x + cos(yaw)*v_body_y
          v_ned_z =  v_body_z  (保持)
        """
        c = math.cos(yaw_rad)
        s = math.sin(yaw_rad)
        return _vec([
            c * v_body[0] - s * v_body[1],
            s * v_body[0] + c * v_body[1],
            v_body[2],
        ])

    def _ned_to_body(self, v_ned: np.ndarray, yaw_rad: float) -> np.ndarray:
        """NED速度 → 机体系速度，用于NED动作模式的日志回显。"""
        c = math.cos(yaw_rad)
        s = math.sin(yaw_rad)
        return _vec([
            c * v_ned[0] + s * v_ned[1],
            -s * v_ned[0] + c * v_ned[1],
            v_ned[2],
        ])


# ============================================================
# 单元测试
# ============================================================

def _test_yaw_zero():
    """yaw=0: 机头朝北 → forward=N, DeepRL action 7=E"""
    dec = ActionDecomposer({})
    v_body, v_ned, yr = dec.decompose(1, 0.0)  # action=1 (forward)
    assert _allclose(v_body, [1, 0, 0]), f"v_body={v_body}"
    assert _allclose(v_ned, [1, 0, 0]), f"v_ned={v_ned}"
    print("  ✅ yaw=0: forward=N → v_ned=[1,0,0]")

    _, v_ned, _ = dec.decompose(7, 0.0)  # action=7 (E in original DeepRL sign)
    assert _allclose(v_ned, [0, 1, 0], atol=1e-6), f"v_ned={v_ned}"
    print("  ✅ yaw=0: DeepRL action 7(E) → v_ned=[0,1,0]")


def _test_yaw_90():
    """yaw=90° (π/2): 机头朝东 → forward=E, DeepRL action 3=W rotates to N"""
    dec = ActionDecomposer({})
    _, v_ned, _ = dec.decompose(1, math.pi / 2)  # forward
    assert _allclose(v_ned, [0, 1, 0], atol=1e-6), f"v_ned={v_ned}"
    print("  ✅ yaw=90°: forward=E → v_ned=[0,1,0]")

    _, v_ned, _ = dec.decompose(3, math.pi / 2)  # W in original DeepRL sign
    assert _allclose(v_ned, [1, 0, 0], atol=1e-6), f"v_ned={v_ned}"
    print("  ✅ yaw=90°: DeepRL action 3(W) → v_ned=[1,0,0]")


def _test_yaw_180():
    """yaw=180° (π): 机头朝南 → forward=S, right=W"""
    dec = ActionDecomposer({})
    _, v_ned, _ = dec.decompose(1, math.pi)  # forward
    assert _allclose(v_ned, [-1, 0, 0], atol=1e-6), f"v_ned={v_ned}"
    print("  ✅ yaw=180°: forward=S → v_ned=[-1,0,0]")


def _test_descend():
    """下降保持 NED down 方向"""
    dec = ActionDecomposer({"vel_vertical": 2.0, "yaw_rate_rad_s": 0.5})
    v_body, v_ned, yr = dec.decompose(9, 1.57)
    assert _allclose(v_body, [0, 0, 2.0]), f"v_body={v_body}"
    assert _allclose(v_ned, [0, 0, 2.0]), f"v_ned={v_ned}"  # 下降不旋转
    assert yr == 0.5
    print("  ✅ descend: z preserved, yaw_rate=0.5")


def _test_hover():
    """悬停全零"""
    dec = ActionDecomposer({})
    v_body, v_ned, yr = dec.decompose(0, 0.5)
    assert _allclose(v_body, [0, 0, 0])
    assert _allclose(v_ned, [0, 0, 0])
    assert yr == 0.0
    print("  ✅ hover: all zeros")


def _test_hover_with_yaw_rate():
    """偏航失控场景: hover 仍保持设定 yaw_rate"""
    dec = ActionDecomposer({"yaw_rate_rad_s": 0.3})
    v_body, v_ned, yr = dec.decompose(0, 0.5)
    assert _allclose(v_body, [0, 0, 0])
    assert _allclose(v_ned, [0, 0, 0])
    assert yr == 0.3
    print("  ✅ hover+yaw: velocity zero, yaw_rate=0.3")


def _test_action_frame_ned():
    """action_frame=ned: 横移动作不随yaw旋转。"""
    dec = ActionDecomposer({"action_frame": "ned"})
    v_body, v_ned, _ = dec.decompose(1, math.pi / 2)  # N in NED, body shows left at yaw=90
    assert _allclose(v_ned, [1, 0, 0], atol=1e-6), f"v_ned={v_ned}"
    assert _allclose(v_body, [0, -1, 0], atol=1e-6), f"v_body={v_body}"
    print("  ✅ action_frame=ned: N action remains world-N at yaw=90°")


def _test_original_deeprl_action_sign():
    """Default action sign matches arch/DeepRL/quadrotor_env.py: vy=-sin(angle)."""
    dec = ActionDecomposer({})
    v_body, _, _ = dec.decompose(3, 0.0)
    assert dec.action_id_to_name(3) == "W"
    assert _allclose(v_body, [0, -1, 0], atol=1e-6), f"v_body={v_body}"
    print("  ✅ DeepRL default: act=3 is W, v_body=[0,-1,0]")


def _test_mirror_e_action_yaw_10deg():
    """Optional +1 mirror reproduces earlier logs: act=3(E) body-right -> NED."""
    dec = ActionDecomposer({"action_lateral_sign": 1})
    v_body, v_ned, _ = dec.decompose(3, math.radians(10.0))
    expected = [-math.sin(math.radians(10.0)), math.cos(math.radians(10.0)), 0.0]
    assert dec.action_id_to_name(3) == "E"
    assert _allclose(v_body, [0, 1, 0], atol=1e-6), f"v_body={v_body}"
    assert _allclose(v_ned, expected, atol=1e-6), f"v_ned={v_ned}, expected={expected}"
    print("  ✅ mirror mode: act=3(E), yaw=10° -> v_ned≈[-0.17,0.98,0]")


def run_tests():
    print("=== ActionDecomposer Unit Tests ===")
    _test_yaw_zero()
    _test_yaw_90()
    _test_yaw_180()
    _test_descend()
    _test_hover()
    _test_hover_with_yaw_rate()
    _test_action_frame_ned()
    _test_original_deeprl_action_sign()
    _test_mirror_e_action_yaw_10deg()
    print("=== ALL PASSED ===\n")


if __name__ == "__main__":
    run_tests()
