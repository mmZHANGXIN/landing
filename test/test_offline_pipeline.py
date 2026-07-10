#!/usr/bin/env python3
"""
离线管线测试 — 用 building/*.npy 点云测试感知+DRL
==================================================
输入:  arch/3.UDPDirect30Hz_cyd_final/points/building/*.npy
输出:  test_output/ 目录下保存每帧的语义图、深度图、BEV安全图
       终端打印每帧动作和完整耗时统计

管线:
  点云(N,3) → HALSS GPU → 语义图(128×128)
            → 深度投影 → 深度图(128×128)
  语义图 + 深度图 → SB3 PPO → 离散动作(0-9)

用法:
  conda activate fylanding
  python test_offline_pipeline.py                    # 跑5帧
  python test_offline_pipeline.py --max_frames 33    # 跑全部
"""

import os, sys, time, argparse, logging
import numpy as np
import yaml

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault("DISPLAY", "")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Test")

from perception import HALSSSafetyEvaluator, DepthProjector, SemanticGenerator
from rl import RLAgent
from control.action_decomposer import ActionDecomposer


def save_img(data, path, mode):
    """保存可视化图片 (matplotlib Agg, 无 GUI)"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    if mode == "semantic":
        rgb = np.zeros((*data.shape, 3), dtype=np.uint8)
        rgb[data == 1] = [0, 200, 0]      # 安全:绿
        rgb[data == 9] = [200, 0, 0]      # 危险:红
        ax.imshow(rgb)
        safe_pct = (data == 1).sum() / data.size * 100
        ax.set_title(f"Semantic (Safe={safe_pct:.0f}%)")
    elif mode == "depth":
        im = ax.imshow(data, cmap='inferno')
        plt.colorbar(im, ax=ax, fraction=0.046, label='m')
        ax.set_title("Depth Map")
    elif mode == "bev":
        ax.imshow(data, cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_title("BEV Safety")
    ax.axis('off')
    fig.tight_layout(pad=0.5)
    fig.savefig(path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="arch/3.UDPDirect30Hz_cyd_final/points/building")
    p.add_argument("--config", default="config/experiment_config.yaml")
    p.add_argument("--max_frames", type=int, default=5)
    p.add_argument("--output_dir", default="test_output")
    args = p.parse_args()

    data_dir = os.path.join(PROJECT_ROOT, args.data_dir)
    config_path = os.path.join(PROJECT_ROOT, args.config)
    out = os.path.join(PROJECT_ROOT, args.output_dir)
    os.makedirs(out, exist_ok=True)

    # ---- 配置 ----
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    pc = cfg["perception"]
    oc = cfg["observation"]
    dc = cfg["decision"]
    uc = cfg["uav"]

    # ---- 初始化模块 ----
    logger.info("=" * 55)
    logger.info(" Initializing pipeline modules...")
    logger.info("=" * 55)

    halss   = HALSSSafetyEvaluator(pc)
    dproj   = DepthProjector(img_width=oc["img_width"], img_height=oc["img_height"],
                             max_range=pc["depth_max_range"])
    sem_gen = SemanticGenerator({**pc, **oc})
    rl      = RLAgent(model_path=os.path.join(PROJECT_ROOT, dc["policy_weights_path"]),
                      img_size=(oc["img_width"], oc["img_height"]),
                      vel_lateral=uc["vel_lateral"], vel_vertical=uc["vel_vertical"])

    # ---- 加载数据 ----
    files = sorted([f for f in os.listdir(data_dir)
                    if f.startswith("point_cloud_data_") and f.endswith(".npy")],
                   key=lambda x: int(x.split("_")[-1].split(".")[0]))
    n_total = len(files)
    n = min(args.max_frames, n_total)

    # 用第一帧估算场景范围
    p0 = np.load(os.path.join(data_dir, files[0]))
    cx, cy = (p0[:, 0].min() + p0[:, 0].max()) / 2, (p0[:, 1].min() + p0[:, 1].max()) / 2
    z_hi, z_lo = p0[:, 2].max() + 3.0, p0[:, 2].min() + 0.5

    logger.info(f"Data: {n_total} files, testing {n}")
    logger.info(f"Scene center: ({cx:.1f}, {cy:.1f})  range Z: {z_lo:.1f}~{z_hi:.1f}m")
    logger.info(f"Output dir: {out}/")
    logger.info("=" * 55)

    # ---- 主循环 ----
    t_all = {"halss": [], "depth": [], "rl": [], "total": []}
    actions = []
    anames = ActionDecomposer(uc).action_names

    for i in range(n):
        t0 = time.perf_counter()

        # 1. 点云
        pts = np.load(os.path.join(data_dir, files[i])).astype(np.float32)
        z = z_hi - (z_hi - z_lo) * i / max(n - 1, 1)
        pose = np.array([cx, cy, z, 0, 0, 0], dtype=np.float32)

        # 2. ROI
        d = np.linalg.norm(pts[:, :2] - pose[:2], axis=1)
        pts_r = pts[d < pc["roi_radius_world"]]

        # 3. HALSS → 语义
        t1 = time.perf_counter()
        r = halss.evaluate(pts_r)
        if r is not None:
            sem = sem_gen.generate(r["bev_data"])
            bev = r["bev_data"]["safe_mesh"].astype(np.float32)
        else:
            sem = np.full((oc["img_height"], oc["img_width"]), pc["danger_class_id"], dtype=np.uint8)
            bev = None
        t_all["halss"].append((time.perf_counter() - t1) * 1000)

        # 4. 深度投影
        t2 = time.perf_counter()
        dep = dproj.project(pts_r, pose)
        t_all["depth"].append((time.perf_counter() - t2) * 1000)

        # 5. RL 推理
        t3 = time.perf_counter()
        act = rl.predict(dep, sem)
        vel = rl.map_action_to_velocity(act)
        t_all["rl"].append((time.perf_counter() - t3) * 1000)
        t_all["total"].append((time.perf_counter() - t0) * 1000)

        actions.append(act)

        # 6. 保存图片
        save_img(sem, f"{out}/f{i:03d}_semantic.png", "semantic")
        save_img(dep, f"{out}/f{i:03d}_depth.png", "depth")
        if bev is not None:
            save_img(bev, f"{out}/f{i:03d}_bev.png", "bev")

        logger.info(
            f"[{i:02d}] act={act}({anames[act]}) vel=({vel[0]:+.1f},{vel[1]:+.1f},{vel[2]:+.1f}) "
            f"z={z:.1f}m | H={t_all['halss'][-1]:.0f}ms D={t_all['depth'][-1]:.0f}ms "
            f"RL={t_all['rl'][-1]:.0f}ms T={t_all['total'][-1]:.0f}ms"
        )

    # ---- 汇总 ----
    logger.info("=" * 55)
    logger.info(" SUMMARY")
    logger.info("=" * 55)

    def avg(x): return sum(x) / len(x) if x else 0
    for k in ["halss", "depth", "rl", "total"]:
        logger.info(f"  {k:8s}: avg={avg(t_all[k]):6.1f}ms  max={max(t_all[k]):6.0f}ms")

    from collections import Counter
    logger.info("  Action distribution:")
    for a, cnt in sorted(Counter(actions).items()):
        logger.info(f"    {a}({anames[a]:3s}): {cnt}/{n} ({cnt/n*100:.0f}%)")

    safe_pcts = []
    for i in range(n):
        try:
            img = plt.imread(f"{out}/f{i:03d}_semantic.png")
        except:
            break
    logger.info(f"\n  ✅ {n * 3} images saved → {out}/")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    main()
