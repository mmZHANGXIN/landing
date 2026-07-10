#!/usr/bin/env python3
"""
在线实时测试 — 订阅 FAST-LIO, 跑原始 HALSS + 深度 + RL
=========================================================
用系统 Python3 (rclpy 可用), 导入 conda 模块 (torch/cv2/RL)
每收到一帧 /cloud_registered → 运行完整管线 → 保存图 + 打印耗时
"""

import sys, os, time, logging, threading, argparse
import numpy as np

# 注入 conda 模块路径
sys.path.insert(0, '/home/orin/evelyn/orin_landing')
sys.path.append('/home/orin/miniconda3/envs/fylanding/lib/python3.8/site-packages')

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Online")

import yaml
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2

from perception import HALSSOriginalEvaluator, DepthProjector
from rl import RLAgent
from control.action_decomposer import ActionDecomposer


class OnlinePipeline(Node):
    def __init__(self, cfg, output_dir, max_frames):
        super().__init__("online_pipeline")

        pc = cfg["perception"]
        oc = cfg["observation"]
        dc = cfg["decision"]
        uc = cfg["uav"]

        self.halss   = HALSSOriginalEvaluator(pc)
        self.dproj   = DepthProjector(img_width=oc["img_width"], img_height=oc["img_height"],
                                      max_range=pc["depth_max_range"])
        self.rl      = RLAgent(model_path=os.path.join(PROJECT_ROOT, dc["policy_weights_path"]),
                               img_size=(oc["img_width"], oc["img_height"]),
                               vel_lateral=uc["vel_lateral"], vel_vertical=uc["vel_vertical"])
        self.roi     = pc["roi_radius_world"]
        self.out     = output_dir
        self.max_n   = max_frames

        # 状态
        self._odom = None
        self._n = 0
        self._t = {"halss": [], "depth": [], "rl": [], "total": []}
        self._acts = []
        self._anames = ActionDecomposer(uc).action_names
        self._lock = threading.Lock()

        # 订阅
        self.create_subscription(Odometry, "/Odometry", self._cb_odom, 10)
        self.create_subscription(PointCloud2, "/cloud_registered", self._cb_pc2, 10)

        logger.info("Online pipeline ready | HALSS=Delaunay | Waiting for data...")

    def _cb_odom(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        r, pitch, y = self._quat2euler(q.x, q.y, q.z, q.w)
        self._odom = np.array([p.x, p.y, p.z, r, pitch, y], dtype=np.float32)

    def _cb_pc2(self, msg):
        if self._n >= self.max_n:
            return

        t0 = time.perf_counter()
        pts = self._parse_pc2(msg)
        if pts is None or len(pts) < 10 or self._odom is None:
            return
        pose = self._odom.copy()

        # ROI
        d = np.linalg.norm(pts[:, :2] - pose[:2], axis=1)
        pts_r = pts[d < self.roi]

        # HALSS
        t1 = time.perf_counter()
        r = self.halss.evaluate(pts_r)
        self._t["halss"].append((time.perf_counter() - t1) * 1000)

        # 深度
        t2 = time.perf_counter()
        dep = self.dproj.project(pts_r, pose)
        self._t["depth"].append((time.perf_counter() - t2) * 1000)

        # 语义 (圆圈→二值)
        import cv2
        if r is not None:
            sem = self._circles_to_binary(r["bev_data"]["circles_raw"])
        else:
            sem = np.full((128, 128), 9, dtype=np.uint8)

        # RL
        t3 = time.perf_counter()
        act = self.rl.predict(dep, sem)
        vel = self.rl.map_action_to_velocity(act)
        self._t["rl"].append((time.perf_counter() - t3) * 1000)
        self._t["total"].append((time.perf_counter() - t0) * 1000)

        self._acts.append(act)
        self._n += 1

        # 保存图
        self._save_images(r, dep, sem, pose, pts_r)

        logger.info(
            f"[{self._n:02d}/{self.max_n}] act={act}({self._anames[act]}) "
            f"vel=({vel[0]:+.1f},{vel[1]:+.1f},{vel[2]:+.1f}) "
            f"H={self._t['halss'][-1]:.0f}ms D={self._t['depth'][-1]:.0f}ms "
            f"RL={self._t['rl'][-1]:.0f}ms T={self._t['total'][-1]:.0f}ms"
        )

        if self._n >= self.max_n:
            self._summary()

    def _parse_pc2(self, msg):
        off = {f.name: f.offset for f in msg.fields}
        if not all(k in off for k in ('x','y','z')):
            return None
        npts = msg.width * msg.height
        if npts == 0: return None
        pts = np.zeros((npts, 3), dtype=np.float32)
        # 用 np.frombuffer 按偏移+步长直接读取
        raw = np.frombuffer(msg.data, dtype=np.float32)
        pts_per_point = msg.point_step // 4  # float32 = 4 bytes
        pts[:, 0] = raw[off['x'] // 4 :: pts_per_point]
        pts[:, 1] = raw[off['y'] // 4 :: pts_per_point]
        pts[:, 2] = raw[off['z'] // 4 :: pts_per_point]
        valid = np.isfinite(pts).all(axis=1)
        return pts[valid] if valid.sum() > 0 else None

    @staticmethod
    def _quat2euler(x, y, z, w):
        import math
        sinr = 2*(w*x + y*z); cosr = 1 - 2*(x*x + y*y)
        roll = math.atan2(sinr, cosr)
        sinp = 2*(w*y - z*x)
        pitch = math.asin(max(-1, min(1, sinp)))
        siny = 2*(w*z + x*y); cosy = 1 - 2*(y*y + z*z)
        yaw = math.atan2(siny, cosy)
        return roll, pitch, yaw

    @staticmethod
    def _circles_to_binary(circles_bgr, h=128, w=128):
        import cv2
        gy = cv2.cvtColor(circles_bgr, cv2.COLOR_BGR2GRAY)
        out = np.full((h, w), 9, dtype=np.uint8)
        sm = cv2.resize(gy, (w, h), interpolation=cv2.INTER_AREA)
        out[sm > 200] = 1
        return out

    def _save_images(self, r, dep, sem, pose, pts_r):
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import cv2
        i = self._n - 1

        # 语义图
        rgb = np.zeros((*sem.shape, 3), dtype=np.uint8)
        rgb[sem==1]=[0,200,0]; rgb[sem==9]=[200,0,0]
        fig, ax = plt.subplots(figsize=(3,3)); ax.imshow(rgb); ax.set_title("Semantic"); ax.axis('off')
        fig.tight_layout(pad=0); fig.savefig(f"{self.out}/f{i:03d}_semantic.png", dpi=100, facecolor='white'); plt.close()

        # 深度图
        fig, ax = plt.subplots(figsize=(3,3)); ax.imshow(dep, cmap='inferno'); ax.set_title("Depth"); ax.axis('off')
        fig.tight_layout(pad=0); fig.savefig(f"{self.out}/f{i:03d}_depth.png", dpi=100, facecolor='white'); plt.close()

        # 原始点云俯视
        fig, ax = plt.subplots(figsize=(3.5,3.5))
        ax.scatter(pts_r[:,0], pts_r[:,1], c=pts_r[:,2], s=0.2, cmap='plasma')
        ax.scatter([pose[0]], [pose[1]], c='red', s=30, marker='x')
        ax.set_title(f"Cloud ({len(pts_r)} pts, z={pose[2]:.1f}m)"); ax.set_aspect('equal'); ax.axis('off')
        fig.tight_layout(pad=0); fig.savefig(f"{self.out}/f{i:03d}_cloud.png", dpi=100, facecolor='white'); plt.close()

        if r is not None:
            bev = r["bev_data"]
            # 高程
            fig, ax = plt.subplots(figsize=(3,3)); ax.imshow(bev["z_mesh"], cmap='plasma'); ax.set_title("Height"); ax.axis('off')
            fig.tight_layout(pad=0); fig.savefig(f"{self.out}/f{i:03d}_height.png", dpi=100, facecolor='white'); plt.close()
            # 法线
            sn = cv2.cvtColor(bev["surf_norm_raw"], cv2.COLOR_BGR2RGB)
            fig, ax = plt.subplots(figsize=(3,3)); ax.imshow(sn); ax.set_title("Surface Normal"); ax.axis('off')
            fig.tight_layout(pad=0); fig.savefig(f"{self.out}/f{i:03d}_norm.png", dpi=100, facecolor='white'); plt.close()

    def _summary(self):
        from collections import Counter
        def avg(x): return sum(x)/len(x) if x else 0
        logger.info("=" * 55)
        logger.info(" ONLINE SUMMARY (%d frames)", self._n)
        logger.info(f"  HALSS:      avg={avg(self._t['halss']):.0f}ms  max={max(self._t['halss']):.0f}ms")
        logger.info(f"  Depth Proj: avg={avg(self._t['depth']):.0f}ms  max={max(self._t['depth']):.0f}ms")
        logger.info(f"  RL Infer:   avg={avg(self._t['rl']):.0f}ms  max={max(self._t['rl']):.0f}ms")
        logger.info(f"  Total:      avg={avg(self._t['total']):.0f}ms  max={max(self._t['total']):.0f}ms")
        logger.info("  Actions:")
        for a, cnt in sorted(Counter(self._acts).items()):
            logger.info(f"    {a}({self._anames[a]:3s}): {cnt}/{self._n}")
        logger.info(f"  Images → {self.out}/")
        rclpy.shutdown()


PROJECT_ROOT = '/home/orin/evelyn/orin_landing'

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max_frames", type=int, default=30)
    p.add_argument("--output_dir", default=os.path.join(PROJECT_ROOT, "test_online"))
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(PROJECT_ROOT, "config/experiment_config.yaml")) as f:
        cfg = yaml.safe_load(f)

    rclpy.init()
    node = OnlinePipeline(cfg, args.output_dir, args.max_frames)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node._summary()


if __name__ == "__main__":
    main()
