#!/usr/bin/env python3
"""
在线管线测试（无飞控）— 订阅 FAST-LIO 实时输出，跑感知+RL
==========================================================
订阅: /Odometry (位姿) + /cloud_registered (去畸变点云)
管线: 点云 → HALSS → 语义图 + 深度图 → DRL → 动作
"""

import os, sys, time, logging
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2

PROJECT_ROOT = "/home/orin/evelyn/orin_landing"
sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault("DISPLAY", "")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("OnlineTest")

from perception import HALSSSafetyEvaluator, DepthProjector, SemanticGenerator
from rl import RLAgent
from control.action_decomposer import ActionDecomposer
# 复用 fastlio_interface 的点云解析
from odometry.fastlio_interface import FastLIOInterface


class OnlineTester(Node):
    def __init__(self):
        super().__init__("online_test")

        # 加载配置
        import yaml
        with open(os.path.join(PROJECT_ROOT, "config/experiment_config.yaml")) as f:
            cfg = yaml.safe_load(f)
        pc = cfg["perception"]
        oc = cfg["observation"]
        dc = cfg["decision"]
        uc = cfg["uav"]

        # 点云接口
        self._lio = FastLIOInterface(use_ros=False)
        # 手动绕开 FastLIOInterface 的反向 ROS init，直接用 numpy 解析
        self._t_pc2 = None
        self._t_odom = None
        import sensor_msgs.msg
        self.PointCloud2 = sensor_msgs.msg.PointCloud2

        # 模块
        self.halss = HALSSSafetyEvaluator(pc)
        self.dproj = DepthProjector(img_width=oc["img_width"], img_height=oc["img_height"],
                                    max_range=pc["depth_max_range"])
        self.semgen = SemanticGenerator({**pc, **oc})
        self.rl = RLAgent(model_path=os.path.join(PROJECT_ROOT, dc["policy_weights_path"]),
                         img_size=(oc["img_width"], oc["img_height"]),
                         vel_lateral=uc["vel_lateral"], vel_vertical=uc["vel_vertical"])
        self.roi = pc["roi_radius_world"]
        self.danger_id = pc["danger_class_id"]

        # 订阅
        self.create_subscription(Odometry, "/Odometry", self._cb_odom, 10)
        self.create_subscription(PointCloud2, "/cloud_registered", self._cb_pc2, 10)

        # 统计
        self._timing = {"halss": [], "depth": [], "rl": [], "total": []}
        self._actions = []
        self._n = 0
        self._max_n = 50
        self._anames = ActionDecomposer(uc).action_names

        logger.info("=" * 55)
        logger.info(" Online tester ready, waiting for FAST-LIO data...")
        logger.info("=" * 55)

    def _cb_odom(self, msg):
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        r, p, y = self._quat2euler(q.x, q.y, q.z, q.w)
        self._t_odom = np.array([pos.x, pos.y, pos.z, r, p, y], dtype=np.float32)

    def _cb_pc2(self, msg):
        """收到点云 → 触发一帧处理"""
        t0 = time.perf_counter()

        if self._t_odom is None:
            return
        if self._n >= self._max_n:
            self._shutdown()
            return

        # 解析点云
        pts = self._parse_pc2(msg)
        if pts is None or len(pts) < 10:
            return
        pose = self._t_odom.copy()

        # ROI
        d = np.linalg.norm(pts[:, :2] - pose[:2], axis=1)
        pts_r = pts[d < self.roi]

        # HALSS
        t1 = time.perf_counter()
        r = self.halss.evaluate(pts_r)
        if r is not None:
            sem = self.semgen.generate(r["bev_data"])
        else:
            sem = np.full((self.rl.img_h, self.rl.img_w), self.danger_id, dtype=np.uint8)
        self._timing["halss"].append((time.perf_counter() - t1) * 1000)

        # 深度投影
        t2 = time.perf_counter()
        dep = self.dproj.project(pts_r, pose)
        self._timing["depth"].append((time.perf_counter() - t2) * 1000)

        # RL
        t3 = time.perf_counter()
        act = self.rl.predict(dep, sem)
        vel = self.rl.map_action_to_velocity(act)
        self._timing["rl"].append((time.perf_counter() - t3) * 1000)
        self._timing["total"].append((time.perf_counter() - t0) * 1000)

        self._actions.append(act)
        self._n += 1

        logger.info(
            f"[{self._n:02d}] act={act}({self._anames[act]}) "
            f"vel=({vel[0]:+.1f},{vel[1]:+.1f},{vel[2]:+.1f}) "
            f"H={self._timing['halss'][-1]:.0f}ms D={self._timing['depth'][-1]:.0f}ms "
            f"RL={self._timing['rl'][-1]:.0f}ms T={self._timing['total'][-1]:.0f}ms"
        )

        if self._n >= self._max_n:
            self._shutdown()

    def _parse_pc2(self, msg):
        """numpy 解析 PointCloud2"""
        try:
            off = {}
            for f in msg.fields:
                off[f.name] = f.offset
            data = np.frombuffer(msg.data, dtype=np.uint8)
            npts = msg.width * msg.height
            if npts == 0:
                return None
            pts = np.zeros((npts, 3), dtype=np.float32)
            for i, name in enumerate(['x', 'y', 'z']):
                pts[:, i] = data[off[name]::msg.point_step].view(np.float32)[:npts]
            valid = np.isfinite(pts).all(axis=1)
            return pts[valid]
        except:
            return None

    @staticmethod
    def _quat2euler(x, y, z, w):
        import math
        sinr_cosp = 2*(w*x + y*z)
        cosr_cosp = 1 - 2*(x*x + y*y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        sinp = 2*(w*y - z*x)
        pitch = math.asin(max(-1, min(1, sinp)))
        siny_cosp = 2*(w*z + x*y)
        cosy_cosp = 1 - 2*(y*y + z*z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

    def _shutdown(self):
        from collections import Counter
        logger.info("=" * 55)
        logger.info(" SUMMARY (%d frames)", self._n)
        logger.info("=" * 55)

        def avg(x): return sum(x)/len(x) if x else 0
        for k in ["halss", "depth", "rl", "total"]:
            logger.info(f"  {k:8s}: avg={avg(self._timing[k]):6.1f}ms  min={min(self._timing[k]):6.0f}ms  max={max(self._timing[k]):6.0f}ms")

        logger.info("  Actions:")
        for a, cnt in sorted(Counter(self._actions).items()):
            logger.info(f"    {a}({self._anames[a]:3s}): {cnt}/{self._n} ({cnt/self._n*100:.0f}%)")

        logger.info("  Done. (no flight control — actions logged only)")
        rclpy.shutdown()


def main():
    rclpy.init()
    node = OnlineTester()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node._shutdown()


if __name__ == "__main__":
    main()
