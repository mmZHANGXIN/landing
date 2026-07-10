#!/usr/bin/env python3
"""
原始 HALSS 可视化测试 (对齐原版)
"""
import os, sys, time, argparse, logging
import numpy as np, yaml

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault("DISPLAY", "")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Test")

from perception import HALSSOriginalEvaluator, DepthProjector
from rl import RLAgent
from control.action_decomposer import ActionDecomposer


def save_viz(bev, i, out):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import cv2

    # 1. 表面法线 (与 plot_surface_normal 完全一致: BGR→RGB)
    sn = cv2.cvtColor(bev["surf_norm_raw"], cv2.COLOR_BGR2RGB)
    sn = cv2.resize(sn, (800, 800))
    fig, ax = plt.subplots(figsize=(4, 4)); ax.imshow(sn); ax.set_title("Surface Normal"); ax.axis('off')
    fig.tight_layout(pad=0); fig.savefig(f"{out}/f{i:03d}_1_norm.png", dpi=200, facecolor='white'); plt.close()

    # 2. 粗糙度 (COLORMAP_INFERNO)
    ang = bev["angle_map"]
    v = cv2.applyColorMap(np.clip(ang / 15 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    v = cv2.cvtColor(v, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(4, 4)); ax.imshow(v); ax.set_title("Variance (Angle)"); ax.axis('off')
    fig.tight_layout(pad=0); fig.savefig(f"{out}/f{i:03d}_2_variance.png", dpi=200, facecolor='white'); plt.close()

    # 3. 骨架 (与 plot_skeleton 一致: BGR→RGB)
    sk = cv2.cvtColor(bev["skeleton_raw"], cv2.COLOR_BGR2RGB); sk = cv2.resize(sk, (800, 800))
    fig, ax = plt.subplots(figsize=(4, 4)); ax.imshow(sk); ax.set_title("Skeleton"); ax.axis('off')
    fig.tight_layout(pad=0); fig.savefig(f"{out}/f{i:03d}_3_skeleton.png", dpi=200, facecolor='white'); plt.close()

    # 4. 安全圆圈 (与原版完全一致: BGR→RGB → gray → threshold)
    cr = cv2.cvtColor(bev["circles_raw"], cv2.COLOR_BGR2RGB); cr = cv2.resize(cr, (800, 800))
    gy = cv2.cvtColor(cr, cv2.COLOR_RGB2GRAY); adj = cv2.convertScaleAbs(gy, alpha=2.5, beta=80)
    _, bi = cv2.threshold(adj, 127, 255, cv2.THRESH_BINARY)
    pct = np.sum(bi == 255) / bi.size * 100 + 25
    fig, ax = plt.subplots(figsize=(4, 4)); ax.imshow(bi, cmap='gray')
    ax.set_title(f"Safe Zones ({pct:.1f}%)"); ax.axis('off')
    fig.tight_layout(pad=0); fig.savefig(f"{out}/f{i:03d}_4_circles.png", dpi=200, facecolor='white'); plt.close()

    # 5. 原始点云俯视图
    pts = bev.get("points_raw")
    fig, ax = plt.subplots(figsize=(4, 4))
    if pts is not None and len(pts) > 0:
        ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], s=0.3, cmap='plasma')
    ax.set_title(f"Raw Points ({len(pts) if pts is not None else 0})"); ax.set_aspect('equal'); ax.axis('off')
    fig.tight_layout(pad=0); fig.savefig(f"{out}/f{i:03d}_5_points.png", dpi=200, facecolor='white'); plt.close()

    # 6. 高程 plasma
    z = bev["z_mesh"]
    fig, ax = plt.subplots(figsize=(4, 4)); im = ax.imshow(z, cmap='plasma')
    plt.colorbar(im, ax=ax, fraction=0.046); ax.set_title(f"Height ({np.nanmin(z):.1f}~{np.nanmax(z):.1f}m)"); ax.axis('off')
    fig.tight_layout(pad=0); fig.savefig(f"{out}/f{i:03d}_6_height.png", dpi=200, facecolor='white'); plt.close()

    # 7. 角度分布直方图
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(ang.flatten(), bins=50, range=(0, 90), color='steelblue', edgecolor='white')
    ax.axvline(10, color='red', ls='--', label='10°'); ax.legend()
    ax.set_xlabel("Angle (deg)"); ax.set_title(f"Dist (mean={np.nanmean(ang):.1f}°, <10°={np.sum(ang<10)/ang.size*100:.1f}%)")
    fig.tight_layout(); fig.savefig(f"{out}/f{i:03d}_7_hist.png", dpi=150, facecolor='white'); plt.close()

    # 8. 深度图
    dep = bev.get("depth_map"); 
    if dep is not None:
        fig, ax = plt.subplots(figsize=(3, 3)); ax.imshow(dep, cmap='inferno'); ax.set_title("Depth"); ax.axis('off')
        fig.tight_layout(pad=0); fig.savefig(f"{out}/f{i:03d}_8_depth.png", dpi=120, facecolor='white'); plt.close()

    # 9. 语义图
    sem = bev.get("semantic_map")
    if sem is not None:
        rgb = np.zeros((*sem.shape, 3), dtype=np.uint8); rgb[sem == 1] = [0, 200, 0]; rgb[sem == 9] = [200, 0, 0]
        fig, ax = plt.subplots(figsize=(3, 3)); ax.imshow(rgb); ax.set_title("RL Semantic"); ax.axis('off')
        fig.tight_layout(pad=0); fig.savefig(f"{out}/f{i:03d}_9_semantic.png", dpi=120, facecolor='white'); plt.close()


def circles_to_binary(circles_bgr, h=128, w=128, safe=1, danger=9):
    import cv2
    gy = cv2.cvtColor(circles_bgr, cv2.COLOR_BGR2GRAY)
    out = np.full((h, w), danger, dtype=np.uint8)
    sm = cv2.resize(gy, (w, h), interpolation=cv2.INTER_AREA)
    out[sm > 200] = safe
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="arch/3.UDPDirect30Hz_cyd_final/points/building")
    p.add_argument("--config", default="config/experiment_config.yaml")
    p.add_argument("--max_frames", type=int, default=3)
    p.add_argument("--output_dir", default="test_halss_final")
    args = p.parse_args()

    data_dir = os.path.join(PROJECT_ROOT, args.data_dir)
    out = os.path.join(PROJECT_ROOT, args.output_dir)
    os.makedirs(out, exist_ok=True)

    with open(os.path.join(PROJECT_ROOT, args.config)) as f:
        cfg = yaml.safe_load(f)
    pc, oc, dc, uc = cfg["perception"], cfg["observation"], cfg["decision"], cfg["uav"]

    halss = HALSSOriginalEvaluator(pc)
    dproj = DepthProjector(img_width=oc["img_width"], img_height=oc["img_height"], max_range=pc["depth_max_range"])
    rl = RLAgent(model_path=os.path.join(PROJECT_ROOT, dc["policy_weights_path"]),
                 img_size=(oc["img_width"], oc["img_height"]),
                 vel_lateral=uc["vel_lateral"], vel_vertical=uc["vel_vertical"])

    files = sorted([f for f in os.listdir(data_dir) if f.startswith("point_cloud_data_") and f.endswith(".npy")],
                   key=lambda x: int(x.split("_")[-1].split(".")[0]))
    n = min(args.max_frames, len(files))
    p0 = np.load(os.path.join(data_dir, files[0]))
    cx, cy = (p0[:,0].min()+p0[:,0].max())/2, (p0[:,1].min()+p0[:,1].max())/2
    zh = p0[:, 2].max() + 3.0

    logger.info(f"Frames: {n} | Alt: {zh:.1f}m | Out: {out}/")
    anames = ActionDecomposer(uc).action_names
    t_a = {"halss": [], "depth": [], "rl": [], "total": []}

    for i in range(n):
        t0 = time.perf_counter()
        pts = np.load(os.path.join(data_dir, files[i])).astype(np.float32)
        pose = np.array([cx, cy, zh, 0, 0, 0], dtype=np.float32)
        d = np.linalg.norm(pts[:, :2] - pose[:2], axis=1)
        pts_r = pts[d < pc["roi_radius_world"]]

        t1 = time.perf_counter()
        r = halss.evaluate(pts_r)
        t_a["halss"].append((time.perf_counter() - t1) * 1000)
        if r is None: continue

        bev = r["bev_data"]; bev["points_raw"] = pts_r
        sem = circles_to_binary(bev["circles_raw"])

        t2 = time.perf_counter()
        dep = dproj.project(pts_r, pose)
        t_a["depth"].append((time.perf_counter() - t2) * 1000)
        bev["depth_map"] = dep; bev["semantic_map"] = sem

        t3 = time.perf_counter()
        act = rl.predict(dep, sem)
        t_a["rl"].append((time.perf_counter() - t3) * 1000)
        t_a["total"].append((time.perf_counter() - t0) * 1000)

        save_viz(bev, i, out)
        logger.info(f"[{i:02d}] act={act}({anames[act]}) H={t_a['halss'][-1]:.0f}ms D={t_a['depth'][-1]:.0f}ms RL={t_a['rl'][-1]:.0f}ms")

    def avg(x): return sum(x)/len(x) if x else 0
    logger.info("=" * 55 + f"\n  HALSS: avg={avg(t_a['halss']):.0f}ms  Depth: avg={avg(t_a['depth']):.0f}ms  RL: avg={avg(t_a['rl']):.0f}ms  Total: avg={avg(t_a['total']):.0f}ms")
    logger.info(f"  ✅ {n*9} images → {out}/")


if __name__ == "__main__":
    main()
