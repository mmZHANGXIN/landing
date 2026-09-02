#!/usr/bin/env python3
"""
从 rosbag 的 /mavros/local_position/odom 生成 Orin Landing 侧视降落轨迹动画
(透明背景), 以真实实验时间 1:1 播放, 可选叠加 DRL 决策面板.

时间轴:
  - 视频起点 = mission_events.csv 中 STAGING_YAW_STARTED.timestamp_ros_s
  - 视频终点 = MANUAL_TAKEOVER.timestamp_ros_s
               (无则回退 SHUTDOWN_REASON, 再无则用 bag 最后有效 odom 时间)
  - 每秒视频 = 一秒真实实验; --fps 只控制采样帧率, 不改变播放速度
  - 每帧叠加 HUD: Time (相对视频时间), ROS (绝对时间戳), Fast-LIO frame
    (按 /cloud_registered_body 输出消息时间排序的连续帧号), Cloud seq
    (该帧 header.seq 及时间差),
    W10 frame (同一云帧在 replay_window10.py 回放中的处理帧号,
    按 --skip-frames 口径复刻其计数, 时间戳最近邻匹配).

用法:
  python3 render_side_trajectory_mov.py <实验目录> [选项]
  python3 render_side_trajectory_mov.py <实验目录> \
      --bag experiments/20260807_162946_orin_landing/input.bag \
      --cloud-topic /cloud_registered_body \
      --cloud-match-tolerance-ms 60 \
      --w10-skip-frames 20 \
      --fps 30

数据来源:
  - mission_events.csv            → 起止 ROS 时间 (STAGING_YAW_STARTED /
                                    MANUAL_TAKEOVER / SHUTDOWN_REASON)
  - input.bag  /mavros/local_position/odom → 轨迹 (header.stamp, ENU x/y/z)
  - input.bag  /cloud_registered_body      → 点云帧 header.seq / header.stamp
  - drl_action_log.csv            → DRL 推理动作与置信度 (可选, ROS 时间戳)

W10 frame 口径:
  复刻 replay_window10.py 主循环的计数: 每个通过稀疏 ROI 检查的云帧
  (fastlio 源, BEV 64×64, 首帧 pose 高度作 ground_z) 依次编号,
  --w10-skip-frames 跳过前 N 帧云 (与 --skip-frames 一致, 默认 20 对应
  Orin 复验口径). 视频窗口内的云帧全部参与计数 (无稀疏跳过), 帧号
  = 该云帧的 window10 处理序号. 若实际 window10 运行参数不同, 调整
  --w10-skip-frames 即可 (跳过帧只整体平移帧号).
"""

import os, sys, argparse, tempfile, shutil
import importlib.util
import types
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch
import rosbag
import rospy
from replay_compare_common import (BagFrameSource, bev_roughness_downsample,
                                   dynamic_roi_half_extents, perception_params,
                                   roi_bounds, world_to_level_body,
                                   _rot_z, _rot_zyx)

# ── 中文字体 ──────────────────────────────────────────────────
_cjk = None
for _fn in ['Heiti TC', 'Hiragino Sans GB', 'PingFang SC', 'STHeiti']:
    if [f for f in fm.fontManager.ttflist if f.name == _fn]:
        _cjk = _fn; break
if _cjk:
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [_cjk] + plt.rcParams['font.sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

# ── 命令行 ────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Render side-view descent trajectory (1:1 real time)')
parser.add_argument('exp_dir', help='Experiment directory')
parser.add_argument('--output', '-o', default=None)
parser.add_argument('--width', type=int, default=1920)
parser.add_argument('--height', type=int, default=1080)
parser.add_argument('--fps', type=int, default=30,
                    help='Video sampling frame rate (does NOT change playback speed)')
parser.add_argument('--bag', default=None,
                    help='Input rosbag path (default: <exp_dir>/input.bag)')
parser.add_argument('--cloud-topic', default='/cloud_registered_body',
                    help='Cloud topic for header.seq overlay')
parser.add_argument('--cloud-match-tolerance-ms', type=float, default=60.0,
                    help='Max |cloud_stamp - video_time| for showing Cloud seq (ms)')
parser.add_argument('--w10-skip-frames', type=int, default=20,
                    help='Skip N cloud frames at bag start when counting window10 '
                         'frame numbers (mirror replay_window10.py --skip-frames; '
                         'default 20 = Orin re-verification runs)')
parser.add_argument('--no-w10-frame', action='store_true',
                    help='Disable W10 frame overlay line')
parser.add_argument('--no-drl-overlay', action='store_true',
                    help='Disable DRL decision panel overlay')
parser.add_argument('--drl-panel-position', default='top-left',
                    choices=['top-left', 'top-right'],
                    help='DRL panel position (default: top-left)')
args = parser.parse_args()

EXP = os.path.abspath(args.exp_dir)
OUT = args.output or os.path.join(EXP, 'trajectory.mov')
W, H = args.width, args.height
FPS = args.fps
CLOUD_TOL_S = args.cloud_match_tolerance_ms / 1000.0


def stamp_to_sec(stamp) -> float:
    return float(stamp.secs) + float(stamp.nsecs) * 1e-9


def short_reason(r) -> str:
    r = str(r)
    if 'offboard' in r.lower():
        return 'OFFBOARD lost'
    return r[:40] + ('…' if len(r) > 40 else '')


# ── W10 frame 映射: 复刻 replay_window10.py 主循环的帧计数 ──────
# body_to_world 与 window10 的私有实现同源 (坐标公式, 改动需保持同步);
# 仅用于复刻"稀疏 ROI 跳过"判定, 与轨迹渲染无关.
def _cfg_vec3(cfg: dict, key: str, default) -> np.ndarray:
    arr = np.asarray(cfg.get(key, default), dtype=np.float32)
    if arr.shape != (3,):
        raise ValueError(f"{key} must be a 3-element vector")
    return arr


def _cfg_mat3(cfg: dict, key: str, default) -> np.ndarray:
    arr = np.asarray(cfg.get(key, default), dtype=np.float32)
    if arr.size != 9:
        raise ValueError(f"{key} must contain 9 values")
    return arr.reshape(3, 3)


def _body_to_world(body_points: np.ndarray, pose: np.ndarray,
                   perc_cfg: dict) -> np.ndarray:
    """与 replay_window10.body_to_world 同源: 点云 → 统一世界坐标 W'."""
    pts = np.asarray(body_points, dtype=np.float32)[:, :3]
    if len(pts) == 0:
        return pts
    r_body_from_imu = _cfg_mat3(perc_cfg, "body_R_from_lidar_imu",
                                [1, 0, 0, 0, 1, 0, 0, 0, 1])
    t_body_from_imu = _cfg_vec3(perc_cfg, "body_T_from_lidar_imu", [0, 0, 0])
    roll, pitch, yaw = float(pose[3]), float(pose[4]), float(pose[5])
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        pts_base = pts @ r_body_from_imu.T + t_body_from_imu
        pts_level = pts_base @ _rot_zyx(roll, pitch, 0.0).T
        pts_level[:, 2] *= -1.0  # ENU up → z-down
        pts_w = pts_level @ _rot_z(yaw).T
    pts_w[:, 0] += float(pose[0])
    pts_w[:, 1] += float(pose[1])
    pts_w[:, 2] -= float(pose[2])
    return pts_w.astype(np.float32, copy=False)


def _import_training_camera():
    """导入 TrainingCameraModel.

    本地 (Mac) 无 torch 时 perception/__init__.py 连带导入失败, 改为
    绕过包初始化直接按文件路径加载 (Orin 上有 torch, 走正常导入).
    """
    try:
        from perception.training_camera_projection import TrainingCameraModel
        return TrainingCameraModel
    except Exception:
        sys.path.insert(0, os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        pkg = types.ModuleType('perception')
        pkg.__path__ = [os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'perception')]
        sys.modules['perception'] = pkg
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for name, path in [('utils.valid_nearest',
                            os.path.join(root, 'utils', 'valid_nearest.py')),
                           ('perception.training_camera_projection',
                            os.path.join(root, 'perception',
                                         'training_camera_projection.py'))]:
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
        return sys.modules['perception.training_camera_projection'].TrainingCameraModel


def _build_w10_frame_map(bag_path: str, exp_dir: str, skip_frames: int):
    """复刻 window10 计数: {cloud_stamp: 处理帧号}.

    逐帧走主循环的稀疏 ROI 检查 (fastlio 源, BEV 64×64, 首帧 pose 高度作
    ground_z, --skip-frames 等价参数), 通过检查的云帧依次编号 1, 2, 3…
    返回 stamp → 帧号 dict; 视频窗口内的云帧全部参与计数 (无稀疏跳过),
    帧号只受 skip_frames 整体平移影响.
    """
    with open(os.path.join(exp_dir, 'experiment_config_snapshot.yaml')) as f:
        cfg = yaml.safe_load(f)
    params = perception_params(cfg)
    camera = _import_training_camera().from_config(
        cfg.get('depth_projection', {}).get('training_camera', {}),
        output_width=params['obs_w'], output_height=params['obs_h'],
        far_m=params['dmax'])
    perc = params['perc_cfg']

    source = BagFrameSource(
        bag_path, cfg, cloud_topic='/cloud_registered_body',
        pose_topic='/mavros/local_position/odom',
        raw_topic='/livox/lidar', imu_topic='/livox/imu',
        cloud_source='fastlio', max_sync_ms=params['max_sync_ms'])
    ground_z = None
    skipped = 0
    frame_count = 0
    wmap = {}
    for frame in source:
        if ground_z is None:
            ground_z = float(frame.pose[2])
        if skipped < skip_frames:
            skipped += 1
            continue
        half_x, half_y = dynamic_roi_half_extents(
            params, float(frame.pose[2]), float(ground_z), camera)
        bounds = roi_bounds(half_x, half_y)
        interp_pose = source.pose_at_interp(frame.cloud_stamp)
        if interp_pose is None:
            interp_pose = frame.pose
        world_points = _body_to_world(frame.cloud_pts, interp_pose, perc)
        level_points = world_to_level_body(world_points, interp_pose)
        bev = bev_roughness_downsample(level_points, bounds, grid_res=64)
        if (not bev.occupied.any()
                or bev.stats['output_points'] < 10):
            continue
        frame_count += 1
        wmap[frame.cloud_stamp] = frame_count
    source.close()
    return wmap


# ── 1. 事件时间轴: STAGING_YAW_STARTED → MANUAL_TAKEOVER ──────
events = pd.read_csv(os.path.join(EXP, 'mission_events.csv'))

def ev(name):
    r = events[events['event'] == name]
    return None if r.empty else r.iloc[0]

st = ev('STAGING_YAW_STARTED')
if st is None:
    print('ERROR: STAGING_YAW_STARTED not found in mission_events.csv'); sys.exit(1)

start_ros = float(st['timestamp_ros_s'])
mt = ev('MANUAL_TAKEOVER')
sr = ev('SHUTDOWN_REASON')

if mt is not None:
    end_ros = float(mt['timestamp_ros_s'])
    end_reason = short_reason(mt['reason'])
    end_src = 'MANUAL_TAKEOVER'
elif sr is not None:
    end_ros = float(sr['timestamp_ros_s'])
    end_reason = short_reason(sr['reason'])
    end_src = 'SHUTDOWN_REASON'
else:
    end_ros = None
    end_reason = 'end of bag'
    end_src = 'last odometry'

print(f'Start (STAGING_YAW_STARTED): t={start_ros:.6f}')
print(f'End   ({end_src}):            t={end_ros:.6f} (reason: {end_reason})')

# ── 2. rosbag: 轨迹 (/mavros/local_position/odom) ─────────────
bag_path = args.bag or os.path.join(EXP, 'input.bag')
if not os.path.exists(bag_path):
    print(f'ERROR: bag not found: {bag_path} (use --bag)'); sys.exit(1)

print(f'\nBag: {bag_path}')
bag = rosbag.Bag(bag_path, 'r')
info = bag.get_type_and_topic_info()
available = set(info.topics.keys())
if '/mavros/local_position/odom' not in available:
    print('ERROR: /mavros/local_position/odom not in bag. Available:',
          ', '.join(sorted(available))); sys.exit(1)

od_t, od_e, od_n, od_u = [], [], [], []
for _, msg, _ in bag.read_messages(topics=['/mavros/local_position/odom']):
    s = stamp_to_sec(msg.header.stamp)
    p = msg.pose.pose.position
    od_t.append(s); od_e.append(p.x); od_n.append(p.y); od_u.append(p.z)

od_t = np.asarray(od_t, dtype=np.float64)
order = np.argsort(od_t, kind='stable')
od_t = od_t[order]
od_e = np.asarray(od_e)[order]; od_n = np.asarray(od_n)[order]; od_u = np.asarray(od_u)[order]
keep = np.r_[True, np.diff(od_t) > 1e-9]
od_t, od_e, od_n, od_u = od_t[keep], od_e[keep], od_n[keep], od_u[keep]
print(f'Odom: {len(od_t)} msgs, {od_t[0]:.3f} → {od_t[-1]:.3f}')

if end_ros is None:
    end_ros = float(od_t[-1])
    print(f'  End fallback: last odometry t={end_ros:.6f}')

if od_t[0] > start_ros or od_t[-1] < end_ros:
    print(f'WARNING: odometry does not cover [start, end] '
          f'({od_t[0]:.3f}..{od_t[-1]:.3f} vs {start_ros:.3f}..{end_ros:.3f})')
    if od_t[0] > start_ros:
        start_ros = float(od_t[0])
        print(f'  Start clamped to first odometry t={start_ros:.6f}')

video_dur = end_ros - start_ros
print(f'Video duration: {video_dur:.2f}s ({video_dur/60.0:.2f} min) @ {FPS}fps')

# ── 3. rosbag: 点云帧 header.stamp / header.seq ───────────────
cloud_stamps = np.empty(0, dtype=np.float64)
cloud_seqs = np.empty(0, dtype=np.int64)
fastlio_frames = np.empty(0, dtype=np.int64)
w10_map = None
if args.cloud_topic not in available:
    print(f'WARNING: cloud topic {args.cloud_topic} not in bag — Cloud seq overlay disabled')
else:
    margin = 2.0  # s, 窗口外扩避免首尾帧最近邻缺失
    t0 = rospy.Time.from_sec(start_ros - margin)
    t1 = rospy.Time.from_sec(end_ros + margin)
    cs_t, cs_s, cs_f = [], [], []
    # 连续帧号按整个 bag 的输出消息计数，不能从视频窗口的 margin 起算。
    global_frame = 0
    for _, msg, _ in bag.read_messages(topics=[args.cloud_topic]):
        stamp = stamp_to_sec(msg.header.stamp)
        global_frame += 1
        if stamp < start_ros - margin or stamp > end_ros + margin:
            continue
        cs_t.append(stamp)
        cs_s.append(int(msg.header.seq))
        cs_f.append(global_frame)
    cs_t = np.asarray(cs_t, dtype=np.float64)
    order = np.argsort(cs_t, kind='stable')
    cs_t = cs_t[order]
    cs_s = np.asarray(cs_s, dtype=np.int64)[order]
    cs_f = np.asarray(cs_f, dtype=np.int64)[order]
    keep = np.r_[True, np.diff(cs_t) > 1e-9]
    cloud_stamps, cloud_seqs = cs_t[keep], cs_s[keep]
    # Fast-LIO 帧号采用整个 bag 中输出云消息的时间顺序（1-based）。
    # 这与 ROS header.seq、W10 的有效处理帧号分别独立。
    fastlio_frames = cs_f[keep]
    print(f'Cloud: {len(cloud_stamps)} frames in [start-2s, end+2s], '
          f'seq {cloud_seqs[0]}..{cloud_seqs[-1]}')

    # W10 frame 映射: 复刻 window10 计数 (--w10-skip-frames 口径)
    if not args.no_w10_frame:
        try:
            w10_map = _build_w10_frame_map(bag_path, EXP, args.w10_skip_frames)
            missing = sum(1 for s in cloud_stamps if s not in w10_map)
            w10_vals = [w10_map[s] for s in cloud_stamps if s in w10_map]
            print(f'W10 frames: {len(w10_map)} processed in bag '
                  f'(skip {args.w10_skip_frames}), {missing} in-window clouds unmapped, '
                  f'window range {min(w10_vals)}..{max(w10_vals)}')
        except Exception as e:
            print(f'WARNING: W10 frame map failed ({e}); showing "--"')

bag.close()

def cloud_at(t: float):
    """t 处最近邻点云帧 → (Fast-LIO帧, header.seq, Δt_ms, W10帧)."""
    if len(cloud_stamps) == 0:
        return None, None, None, None
    j = int(np.searchsorted(cloud_stamps, t, side='left'))
    best_j, best_d = None, float('inf')
    if j < len(cloud_stamps):
        best_j, best_d = j, abs(cloud_stamps[j] - t)
    if j > 0:
        d = abs(cloud_stamps[j - 1] - t)
        if d < best_d:
            best_j, best_d = j - 1, d
    if best_d > CLOUD_TOL_S:
        return None, None, None, None
    fastlio_frame = int(fastlio_frames[best_j])
    seq = int(cloud_seqs[best_j])
    w10 = int(w10_map.get(cloud_stamps[best_j], 0)) if w10_map is not None else 0
    return fastlio_frame, seq, float((cloud_stamps[best_j] - t) * 1000.0), w10

# ── 4. 1:1 重采样: 每帧恰一个采样点, 帧 i 时间 = start + i/fps ──
n_frames = int(round(video_dur * FPS))
t_rs = start_ros + np.arange(n_frames, dtype=np.float64) / FPS
e_rs = np.interp(t_rs, od_t, od_e)
n_rs = np.interp(t_rs, od_t, od_n)
u_rs = np.interp(t_rs, od_t, od_u)
traj_dur = video_dur

print(f'\nTrajectory: {len(t_rs)} frames over {traj_dur:.2f}s')
print(f'  Start ENU=({e_rs[0]:.3f},{n_rs[0]:.3f},{u_rs[0]:.3f}) '
      f'End ENU=({e_rs[-1]:.3f},{n_rs[-1]:.3f},{u_rs[-1]:.3f})')
print(f'  Height drop: {u_rs[0]-u_rs[-1]:.1f}m')

# ── 5. DRL 动作日志 (ROS 时间戳直接匹配) ───────────────────────
drl_active = False
drl_path = os.path.join(EXP, 'drl_action_log.csv')
if args.no_drl_overlay:
    print('\nDRL overlay: disabled by --no-drl-overlay')
elif not os.path.exists(drl_path):
    print('\nDRL overlay: no drl_action_log.csv found')
else:
    drl_raw = pd.read_csv(drl_path)
    n_total = len(drl_raw)
    print(f'\nDRL actions: {n_total} records in log')

    mapped = []
    skipped = 0
    for _, row in drl_raw.iterrows():
        try:
            probs = np.array([float(x) for x in str(row['action_probs']).split(',')])
        except (ValueError, AttributeError):
            skipped += 1; continue
        aid = int(row['action_id'])
        if 0 <= aid < len(probs):
            conf = float(probs[aid])
        else:
            conf = float(probs.max())
        mapped.append((float(row['timestamp_ros_s']),
                       str(row['action_name']), conf))

    mapped.sort(key=lambda x: x[0])
    if mapped:
        drl_times = np.array([m[0] for m in mapped])
        drl_names = [m[1] for m in mapped]
        drl_confs = np.array([m[2] for m in mapped])
        drl_first_t = drl_times[0]
        drl_last_t = drl_times[-1]

        unique_actions = {}
        for n in drl_names:
            unique_actions[n] = unique_actions.get(n, 0) + 1

        drl_stats = {
            'total_in_log': n_total, 'mapped': len(mapped),
            'skipped': skipped,
            'time_range': f'{drl_first_t:.1f}s → {drl_last_t:.1f}s',
            'actions': unique_actions,
        }

        def get_drl_at(t):
            """视频 ROS 时间 → 最近一条已产生的动作 (action_name, conf); 范围外 (None, 0)."""
            if t < drl_first_t or t > drl_last_t:
                return None, 0.0
            idx = int(np.searchsorted(drl_times, t, side='right')) - 1
            if idx < 0:
                return None, 0.0
            return drl_names[idx], float(drl_confs[idx])

        drl_active = True
        print(f'  Mapped: {drl_stats["mapped"]}, skipped: {drl_stats["skipped"]}')
        print(f'  Time range: {drl_stats["time_range"]}')
        print(f'  Actions: {drl_stats["actions"]}')
    else:
        print('  No valid DRL actions mapped (all skipped)')

if not drl_active and not args.no_drl_overlay and os.path.exists(drl_path):
    print('  DRL panel: inactive (no valid mappings)')

# ── 6. 侧视投影 ──────────────────────────────────────────────
dx = e_rs[-1] - e_rs[0]; dy = n_rs[-1] - n_rs[0]
proj_len = np.sqrt(dx**2 + dy**2)
if proj_len < 1e-6: proj_len = 1.0; dx, dy = 1.0, 0.0
ux, uy = dx/proj_len, dy/proj_len

h_dist = (e_rs - e_rs[-1])*ux + (n_rs - n_rs[-1])*uy
heights = u_rs - u_rs[-1]
total_h = abs(h_dist[-1] - h_dist[0])

print(f'\nProjection: {proj_len:.2f}m start→end horizontal')
print(f'  Start h={h_dist[0]:.2f}m v={heights[0]:.1f}m')
print(f'  End   h={h_dist[-1]:.2f}m v={heights[-1]:.1f}m')

# ── 7. 绘图 ──────────────────────────────────────────────────
dpi = 100
fig = plt.figure(figsize=(W/dpi, H/dpi), dpi=dpi,
                facecolor='none', edgecolor='none')
ax = fig.add_axes([0.08, 0.12, 0.88, 0.80], facecolor='none')

h_margin = max(total_h*0.1, 3)
v_margin = max(heights.max()*0.08, 3)
h_min = h_dist.min() - h_margin; h_max = h_dist.max() + h_margin
v_min = -v_margin; v_max = heights.max() + v_margin

plot_w = W*0.88; plot_h = H*0.80
if (h_max-h_min)/(v_max-v_min) < plot_w/plot_h:
    need = (v_max-v_min)*plot_w/plot_h
    h_min -= (need-(h_max-h_min))/2; h_max += (need-(h_max-h_min))/2
else:
    need = (h_max-h_min)/plot_w*plot_h
    v_min -= (need-(v_max-v_min))/2; v_max += (need-(v_max-v_min))/2

ax.set_xlim(h_min, h_max); ax.set_ylim(v_min, v_max)
ax.set_aspect('equal'); ax.axis('off')

# 起止标记
ax.plot(h_dist[0], heights[0], 'o', color='#00FF88', markersize=14,
        mec='white', mew=2, alpha=0.95, zorder=30)
ax.annotate('Staging Yaw\nStart', (h_dist[0], heights[0]),
            xytext=(14,14), textcoords='offset points',
            color='#00FF88', fontsize=12, fontweight='bold', alpha=0.9, va='bottom')
ax.plot(h_dist[-1], heights[-1], 's', color='#FF8800', markersize=10,
        mec='white', mew=1.5, alpha=0.95, zorder=30)
ax.annotate(f'Endpoint ({end_reason})', (h_dist[-1], heights[-1]),
            xytext=(10,-22), textcoords='offset points',
            color='#FF8800', fontsize=12, fontweight='bold', alpha=0.9, va='top')

# 动态元素
traj_line, = ax.plot([],[], '-', color='#00E5FF', alpha=0.9, lw=3.5,
                      solid_capstyle='round', zorder=20)
aircraft, = ax.plot([],[], 'o', color='white', ms=9,
                     mec='#00E5FF', mew=2, alpha=1.0, zorder=40)

# 文字 HUD (右上: 飞行信息; 左下: 时间/ROS/Cloud seq)
txt_title = ax.text(0.99, 0.98, 'Orin Landing · Descent Phase',
                    transform=ax.transAxes, fontsize=16, color='white',
                    alpha=0.9, fontweight='bold', ha='right', va='top',
                    fontfamily='sans-serif')
txt_info = ax.text(0.99, 0.92, '', transform=ax.transAxes,
                   fontsize=11, color='white', alpha=0.65,
                   fontfamily='monospace', ha='right', va='top', linespacing=1.6)
txt_time = ax.text(0.01, 0.02, '', transform=ax.transAxes,
                   fontsize=13, color='white', alpha=0.75,
                   fontfamily='monospace', va='bottom', linespacing=1.6)

# ── 8. DRL 面板 (静态元素) ────────────────────────────────────
panel_x = 0.025 if args.drl_panel_position == 'top-left' else 0.72
panel_y = 0.82
panel_w = 0.25
panel_h = 0.13

panel_rect = None
drl_title_txt = None
drl_action_txt = None
drl_conf_bar_bg = None
drl_conf_bar = None
drl_conf_txt = None

if drl_active:
    panel_rect = FancyBboxPatch(
        (panel_x, panel_y), panel_w, panel_h,
        boxstyle='round,pad=0.012', transform=ax.transAxes,
        facecolor=(1, 1, 1, 0.96),
        edgecolor=(0.20, 0.20, 0.20, 0.90), linewidth=1.5, zorder=100)
    ax.add_patch(panel_rect)

    drl_title_txt = ax.text(
        panel_x + 0.015, panel_y + panel_h - 0.025,
        'DRL Policy', transform=ax.transAxes,
        fontsize=14, color='#00E5FF', alpha=0.95,
        fontweight='bold', fontfamily='sans-serif', va='top', zorder=103)

    drl_action_txt = ax.text(
        panel_x + 0.015, panel_y + panel_h - 0.055,
        '', transform=ax.transAxes,
        fontsize=13, color='black', alpha=1.0,
        fontweight='bold', fontfamily='sans-serif', va='top', zorder=103)

    drl_conf_txt = ax.text(
        panel_x + 0.015, panel_y + panel_h - 0.078,
        '', transform=ax.transAxes,
        fontsize=11, color='black', alpha=1.0,
        fontfamily='sans-serif', va='top', zorder=103)

    bar_x = panel_x + 0.015; bar_y = panel_y + 0.018
    bar_w = panel_w - 0.03; bar_h = 0.02
    drl_conf_bar_bg = FancyBboxPatch(
        (bar_x, bar_y), bar_w, bar_h,
        boxstyle='round,pad=0.004', transform=ax.transAxes,
        facecolor=(0.20, 0.20, 0.20, 1.0), edgecolor='none', zorder=101)
    ax.add_patch(drl_conf_bar_bg)

    drl_conf_bar = FancyBboxPatch(
        (bar_x, bar_y), 0.001, bar_h,
        boxstyle='round,pad=0.004', transform=ax.transAxes,
        facecolor='#00E5FF', edgecolor='none', alpha=0.9, zorder=102)
    ax.add_patch(drl_conf_bar)

# ── 9. 帧渲染 ────────────────────────────────────────────────
tmpdir = tempfile.mkdtemp(prefix='traj_side_')
print(f'\nRendering {tmpdir} ...')

matched = 0
dt_min, dt_max = float('inf'), -float('inf')
seq_decreases = 0
w10_decreases = 0
last_seq = -1
last_w10 = -1
w10_first = -1

for fi in range(n_frames):
    t_real = start_ros + fi / FPS
    assert abs(t_real - t_rs[fi]) < 1e-9, 'frame time must equal start + i/fps'
    idx = fi

    # 轨迹 + 航机
    traj_line.set_data(h_dist[:idx+1], heights[:idx+1])
    h_cur, v_cur = h_dist[idx], heights[idx]
    aircraft.set_data([h_cur], [v_cur])

    # 点云最近邻 (Fast-LIO连续帧号 + ROS header.seq + W10帧号)
    fastlio_frame, seq, dms, w10 = cloud_at(t_real)
    if fastlio_frame is not None:
        matched += 1
        dt_min = min(dt_min, dms); dt_max = max(dt_max, dms)
        if seq < last_seq:
            seq_decreases += 1
        last_seq = seq
        fastlio_line = f'Fast-LIO frame: {fastlio_frame}'
        cloud_line = f'Cloud seq: {seq}  (Δt={dms:+.1f} ms)'
        if w10 > 0:
            if w10 < last_w10:
                w10_decreases += 1
            if w10_first < 0:
                w10_first = w10
            last_w10 = w10
            w10_line = f'W10 frame: {w10}'
        else:
            w10_line = 'W10 frame: --'
    else:
        fastlio_line = 'Fast-LIO frame: --'
        cloud_line = 'Cloud seq: --'
        w10_line = 'W10 frame: --'

    # 文字: Time / ROS / Fast-LIO frame / Cloud seq (/ W10 frame)
    m, s = divmod(t_real - start_ros, 60)
    hud = (f'Time: {int(m):02d}:{s:05.2f}\n'
           f'ROS: {t_real:.3f}\n'
           f'{fastlio_line}\n'
           f'{cloud_line}')
    if not args.no_w10_frame:
        hud += f'\n{w10_line}'
    txt_time.set_text(hud)
    txt_info.set_text(
        f'Altitude       {v_cur:5.1f} m\n'
        f'To end         {abs(h_dist[-1]-h_cur):5.1f} m\n'
        f'H dist total   {total_h:5.1f} m')

    # DRL 面板更新 (日志时间范围外隐藏)
    if drl_active:
        action, conf = get_drl_at(t_real)
        if action is not None:
            panel_rect.set_visible(True)
            drl_title_txt.set_visible(True)
            drl_action_txt.set_text(f'Action: {action}')
            drl_action_txt.set_visible(True)
            conf_bar_w = (panel_w - 0.03) * max(0.0, min(1.0, conf))
            drl_conf_bar.set_width(conf_bar_w)
            drl_conf_bar.set_visible(True)
            drl_conf_bar_bg.set_visible(True)
            drl_conf_txt.set_text(f'Confidence: {conf*100:.0f}%')
            drl_conf_txt.set_visible(True)
        else:
            panel_rect.set_visible(False)
            drl_title_txt.set_visible(False)
            drl_action_txt.set_visible(False)
            drl_conf_bar.set_visible(False)
            drl_conf_bar_bg.set_visible(False)
            drl_conf_txt.set_visible(False)

    fig.savefig(os.path.join(tmpdir, f'frame_{fi:05d}.png'),
                dpi=dpi, facecolor='none', edgecolor='none',
                transparent=True, pad_inches=0)

    if (fi+1) % 120 == 0:
        print(f'  {fi+1}/{n_frames} frames')

print(f'All {n_frames} frames rendered.')
print(f'Cloud seq: matched {matched}/{n_frames} '
      f'({100.0*matched/n_frames:.1f}%), '
      f'Δt {dt_min:+.1f}..{dt_max:+.1f} ms, '
      f'non-monotonic {seq_decreases}')
if w10_map is not None:
    print(f'W10 frame: matched {matched}/{n_frames}, '
          f'range {w10_first}..{last_w10}, '
          f'non-monotonic {w10_decreases}')

# ── 10. FFmpeg ───────────────────────────────────────────────
print(f'\nEncoding {OUT} ...')
cmd = (f'ffmpeg -y -framerate {FPS} '
       f'-i "{tmpdir}/frame_%05d.png" '
       f'-c:v prores_ks -profile:v 4444 -pix_fmt yuva444p10le '
       f'-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
       f'"{OUT}" 2>&1')
ret = os.system(cmd)
if ret != 0:
    print('ProRes failed, trying PNG codec...')
    ret = os.system(f'ffmpeg -y -framerate {FPS} '
                    f'-i "{tmpdir}/frame_%05d.png" '
                    f'-c:v png -pix_fmt rgba "{OUT}" 2>&1')
    if ret != 0:
        print(f'\n⚠️  FFmpeg failed. Frames: {tmpdir}'); sys.exit(1)

shutil.rmtree(tmpdir, ignore_errors=True)

mb = os.path.getsize(OUT)/(1024*1024)
print(f'\n✅ {OUT}')
print(f'   {W}×{H} @ {FPS}fps, {video_dur:.2f}s real time (1:1), '
      f'ProRes 4444 + Alpha, {mb:.1f} MB')
print(f'   Range: {start_ros:.3f} → {end_ros:.3f} (ROS)')
print(f'   Cloud: matched {matched}/{n_frames}, tol {args.cloud_match_tolerance_ms} ms, '
      f'Δt {dt_min:+.1f}..{dt_max:+.1f} ms')
if w10_map is not None:
    print(f'   W10 frame: {w10_first}..{last_w10} (skip {args.w10_skip_frames} 口径)')
if drl_active:
    print(f'   DRL panel: {drl_stats["mapped"]} actions, '
          f'{len(drl_stats["actions"])} unique types')
