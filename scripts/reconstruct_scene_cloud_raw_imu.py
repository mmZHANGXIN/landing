#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用原始 Livox CustomMsg + IMU 离线去畸变并重建场景地图。

每个 Livox 点带有相对扫描起点的 ``offset_time``（ns）。脚本用 IMU
角速度积分得到扫描内相对旋转，并用同一时刻的 PX4 odometry 提供平移
补偿和扫描参考位姿。这样不读取 ``/cloud_registered_body``，可独立验证
原始输入经过 IMU deskew 后的地图质量。

输出默认写入 ``scene_map_raw_imu/``，不会覆盖已有 FAST-LIO 地图。
依赖：rosbag、numpy、scipy、pyyaml。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from reconstruct_scene_cloud import (
    VoxelMap, _cfg_mat3, _cfg_vec3, interpolate_pose, load_config,
    odom_to_pose6, parse_pc2, stamp_to_sec, to_world, write_pcd, write_ply,
)

log = logging.getLogger("RawImuMap")


def parse_raw(msg):
    """CustomMsg -> xyz, intensity, per-point absolute timestamp."""
    n = min(int(getattr(msg, "point_num", len(msg.points))), len(msg.points))
    if n == 0:
        return (np.empty((0, 3), np.float32), np.empty(0, np.float32),
                np.empty(0, np.float64))
    p = msg.points[:n]
    xyz = np.asarray([(x.x, x.y, x.z) for x in p], dtype=np.float32)
    intensity = np.asarray([x.reflectivity for x in p], dtype=np.float32)
    # livox_ros_driver2 可将一帧固定为 19968 点，尾部/无效点以零填充。
    # 与 FAST-LIO preprocess.cpp 保持一致，仅保留有效 tag 和非盲区点。
    tags = np.asarray([x.tag for x in p], dtype=np.uint8)
    tag_class = tags & 0x30
    valid_tag = (tag_class == 0x00) | (tag_class == 0x10)
    valid_range = np.einsum("ij,ij->i", xyz, xyz) > 0.1 * 0.1
    # Livox CustomPoint.offset_time is nanoseconds from msg.header.stamp.
    offsets = np.asarray([x.offset_time for x in p], dtype=np.float64) * 1e-9
    finite = (np.isfinite(xyz).all(axis=1) & np.isfinite(offsets) &
              valid_tag & valid_range)
    return xyz[finite], intensity[finite], offsets[finite]


def imu_rotation_table(bag, topic):
    ts, gyro = [], []
    for _, msg, _ in bag.read_messages(topics=[topic]):
        ts.append(stamp_to_sec(msg.header.stamp))
        gyro.append((msg.angular_velocity.x, msg.angular_velocity.y,
                     msg.angular_velocity.z))
    ts = np.asarray(ts, np.float64)
    gyro = np.asarray(gyro, np.float64)
    if len(ts) < 2:
        raise RuntimeError("/livox/imu 中没有足够的 IMU 样本")
    order = np.argsort(ts, kind="stable")
    ts, gyro = ts[order], gyro[order]
    keep = np.r_[True, np.diff(ts) > 1e-9]
    ts, gyro = ts[keep], gyro[keep]
    # q[k] maps coordinates at the first IMU sample to coordinates at k.
    q = [Rotation.identity()]
    for i in range(1, len(ts)):
        dt = float(np.clip(ts[i] - ts[i - 1], 0.0, 0.05))
        q.append(q[-1] * Rotation.from_rotvec(gyro[i - 1] * dt))
    return ts, gyro, Rotation.concatenate(q)


def relative_imu_rotation(imu_t, imu_q, times, ref_t):
    """Return R(ref <- point) for each point time, nearest integrated IMU pose."""
    idx = np.searchsorted(imu_t, times, side="left").clip(0, len(imu_t) - 1)
    left = np.maximum(idx - 1, 0)
    choose_left = np.abs(imu_t[left] - times) < np.abs(imu_t[idx] - times)
    idx[choose_left] = left[choose_left]
    ir = int(np.argmin(np.abs(imu_t - ref_t)))
    r_ref = imu_q[ir].as_matrix()
    r_point = imu_q[idx].as_matrix()
    return np.einsum("ij,njk->nik", r_ref.T, r_point)


def main():
    ap = argparse.ArgumentParser(description="原始 Livox + IMU 去畸变场景重建")
    ap.add_argument("--bag", default="experiments/20260807_162946_orin_landing/input.bag")
    ap.add_argument("--config", default=None)
    ap.add_argument("--raw-topic", default="/livox/lidar")
    ap.add_argument("--imu-topic", default="/livox/imu")
    ap.add_argument("--pose-topic", default="/mavros/local_position/odom")
    ap.add_argument("--voxel-size", type=float, default=0.05)
    ap.add_argument("--max-sync-ms", type=float, default=100.0)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    bag_path = Path(args.bag)
    config_path = Path(args.config) if args.config else bag_path.parent / "experiment_config_snapshot.yaml"
    out = Path(args.output_dir) if args.output_dir else bag_path.parent / "scene_map_raw_imu"
    if not bag_path.is_file() or not config_path.is_file():
        sys.exit(f"bag/config not found: {bag_path} / {config_path}")
    cfg = load_config(str(config_path)); perc = cfg.get("perception", {})
    r_bl = _cfg_mat3(perc, "body_R_from_lidar_imu", np.eye(3))
    t_bl = _cfg_vec3(perc, "body_T_from_lidar_imu", np.zeros(3))
    import rosbag
    bag = rosbag.Bag(str(bag_path), "r")
    info = bag.get_type_and_topic_info().topics
    for topic in (args.raw_topic, args.imu_topic, args.pose_topic):
        if topic not in info:
            sys.exit(f"topic missing: {topic}; available: {', '.join(sorted(info))}")
    imu_t, _, imu_q = imu_rotation_table(bag, args.imu_topic)
    poses, quats, pose_t = [], [], []
    for _, msg, _ in bag.read_messages(topics=[args.pose_topic]):
        p, q = odom_to_pose6(msg)
        if p is not None: pose_t.append(stamp_to_sec(msg.header.stamp)); poses.append(p); quats.append(q)
    pose_t, poses, quats = map(np.asarray, (pose_t, poses, quats))
    if not len(pose_t): sys.exit("no valid odometry samples")
    voxel = VoxelMap(args.voxel_size, with_intensity=True)
    frames = processed = points_in = 0; sync = []; t0 = time.perf_counter()
    for _, msg, _ in bag.read_messages(topics=[args.raw_topic]):
        if args.max_frames and processed >= args.max_frames: break
        xyz, intensity, offsets = parse_raw(msg); frames += 1
        if not len(xyz): continue
        stamp = stamp_to_sec(msg.header.stamp); ref_t = stamp + float(offsets.max())
        ref_pose, err, status = interpolate_pose(pose_t, poses, quats, ref_t, args.max_sync_ms / 1000)
        if ref_pose is None: continue
        point_t = stamp + offsets
        # IMU rotation is expressed in IMU coordinates; conjugate into body.
        r_imu = relative_imu_rotation(imu_t, imu_q, point_t, ref_t)
        r_body = np.einsum("ij,njk,kl->nil", r_bl, r_imu, r_bl.T)
        p_body = xyz @ r_bl.T + t_bl
        p_ref = np.einsum("nij,nj->ni", r_body, p_body)
        # Translation uses synchronized odometry only for displacement; rotation
        # within the scan remains the IMU-derived rotation above.
        point_poses = []
        for t in point_t[::max(1, len(point_t)//256)]:
            pp, _, _ = interpolate_pose(pose_t, poses, quats, float(t), args.max_sync_ms / 1000)
            point_poses.append(pp)
        if point_poses and all(p is not None for p in point_poses):
            sample_t = point_t[::max(1, len(point_t)//256)]
            d = np.asarray(point_poses)[:, :3]
            delta = np.column_stack([np.interp(point_t, sample_t, d[:, k]) for k in range(3)]) - ref_pose[:3]
            r_ref = to_world(np.zeros((1, 3), np.float32), ref_pose, np.eye(3, dtype=np.float32), np.zeros(3))
            # Convert world displacement to ref-body coordinates, preserving
            # the exact reference pose used for map insertion.
            from reconstruct_scene_cloud import _rot_zyx
            delta_ref = delta @ _rot_zyx(*map(float, ref_pose[3:])).astype(np.float32)
            p_ref += delta_ref
        pts_world = to_world(p_ref, ref_pose, np.eye(3, dtype=np.float32), np.zeros(3))
        points_in += voxel.add(pts_world, intensity); processed += 1; sync.append(err * 1000)
        if processed % 100 == 0: log.info("frames=%d points=%d voxels=%d", processed, points_in, voxel.occupied_voxels)
    bag.close(); pts, ints, _ = voxel.finalize(); out.mkdir(parents=True, exist_ok=True)
    write_ply(out / "scene_map.ply", pts, ints); write_pcd(out / "scene_map.pcd", pts, ints)
    stats = {"source": "raw Livox CustomMsg + /livox/imu", "raw_topic": args.raw_topic,
             "imu_topic": args.imu_topic, "pose_topic": args.pose_topic,
             "deskew": "gyro_integrated_rotation + odometry_translation",
             "voxel_size_m": args.voxel_size, "frames": {"total": frames, "processed": processed},
             "input_points": points_in, "output_points": len(pts),
             "pose_sync_error_ms": {"mean": float(np.mean(sync)) if sync else None,
                                    "p95": float(np.percentile(sync, 95)) if sync else None},
             "elapsed_s": time.perf_counter() - t0}
    (out / "scene_map_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    log.info("done: %d frames, %d points -> %s", processed, len(pts), out)


if __name__ == "__main__": main()
