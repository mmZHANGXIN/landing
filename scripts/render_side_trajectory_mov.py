#!/usr/bin/env python3
"""
从 PX4 ULog 生成 Orin Landing 侧视降落轨迹动画 (透明背景).

用法:
  python3 render_side_trajectory_mov.py <实验目录> [选项]

范围: GOTO_ARRIVED (安全点抵达) → PX4 首次稳定 landed=true
视角: 侧视 — 水平轴沿安全点→接地点地面投影方向, 纵轴为相对接地点高度 (接地=0m)

数据来源:
  - mission_events.csv        → GOTO_ARRIVED 事件 ENU 坐标
  - experiment_config_snapshot.yaml → 配置安全点 GPS (仅用于信息输出)
  - *.ulg                     → vehicle_local_position + vehicle_land_detected
"""

import os, sys, argparse, tempfile, shutil
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pyulog import ULog

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
parser = argparse.ArgumentParser(description='Render side-view descent trajectory')
parser.add_argument('exp_dir', help='Experiment directory')
parser.add_argument('--output', '-o', default=None)
parser.add_argument('--width', type=int, default=1920)
parser.add_argument('--height', type=int, default=1080)
parser.add_argument('--fps', type=int, default=30)
parser.add_argument('--duration', type=float, default=12.0)
args = parser.parse_args()

EXP = os.path.abspath(args.exp_dir)
OUT = args.output or os.path.join(EXP, 'trajectory.mov')
W, H = args.width, args.height
FPS = args.fps
DUR = args.duration

# ── 1. 读取 GOTO_ARRIVED 事件 ENU ────────────────────────────
events = pd.read_csv(os.path.join(EXP, 'mission_events.csv'))

def ev(name):
    r = events[events['event'] == name]
    return None if r.empty else r.iloc[0]

ga = ev('GOTO_ARRIVED')
if ga is None:
    print('ERROR: GOTO_ARRIVED not found')
    sys.exit(1)

GA_ENU = np.array([ga['enu_x'], ga['enu_y'], ga['enu_z']], dtype=float)
print(f'GOTO_ARRIVED ENU: E={GA_ENU[0]:.3f}, N={GA_ENU[1]:.3f}, U={GA_ENU[2]:.3f}')

# ── 2. 读取配置安全点 (仅信息输出) ──────────────────────────
with open(os.path.join(EXP, 'experiment_config_snapshot.yaml')) as f:
    config = yaml.safe_load(f)
gp = config['global_prior']
print(f'Config safe GPS: {gp["target_lat"]:.7f}, {gp["target_lon"]:.7f}')

# ── 3. 读取 ULog ─────────────────────────────────────────────
ulg_files = [f for f in os.listdir(EXP) if f.endswith('.ulg')]
if not ulg_files:
    print('ERROR: No .ulg found')
    sys.exit(1)
ulg_path = os.path.join(EXP, ulg_files[0])
print(f'ULog: {os.path.basename(ulg_path)}')

ulog = ULog(ulg_path)

# vehicle_local_position: NED (x=N, y=E, z=D)
lp = ulog.get_dataset('vehicle_local_position')
ts_lp  = lp.data['timestamp'] / 1e6
e_enu  = lp.data['y']           # NED y → ENU E
n_enu  = lp.data['x']           # NED x → ENU N
u_enu  = -lp.data['z']          # NED z (down) → ENU U (up)

# vehicle_land_detected
ld = ulog.get_dataset('vehicle_land_detected')
ts_ld  = ld.data['timestamp'] / 1e6
landed = ld.data['landed']

# ── 4. 起点: 匹配 GOTO_ARRIVED ENU 最近的 ULog 样本 ─────────
dist = np.sqrt(
    (e_enu - GA_ENU[0])**2 +
    (n_enu - GA_ENU[1])**2 +
    (u_enu - GA_ENU[2])**2
)
start_idx = int(np.argmin(dist))
start_err = float(dist[start_idx])

print(f'\nStart match: idx={start_idx}, ULog t={ts_lp[start_idx]:.3f}s')
print(f'  ULog ENU: E={e_enu[start_idx]:.3f}, N={n_enu[start_idx]:.3f}, U={u_enu[start_idx]:.3f}')
print(f'  Match error: {start_err:.4f}m')

if start_err > 0.5:
    print(f'ERROR: ENU match error {start_err:.3f}m > 0.5m threshold')
    print('  Coordinate frames may differ between mission_events and ULog')
    sys.exit(1)

# ── 5. 终点: 起点之后首次连续 ≥0.5s 的 landed=true ──────────
MIN_LANDED_S = 0.5
land_ulog_t = None

for i in range(len(landed)):
    if landed[i] != 1 or ts_ld[i] <= ts_lp[start_idx]:
        continue
    # 计算从 i 开始的连续 landed=true 时长
    cons = 1
    for j in range(i + 1, len(landed)):
        if landed[j] == 1:
            cons += 1
        else:
            break
    dur = ts_ld[i + cons - 1] - ts_ld[i]
    if dur >= MIN_LANDED_S:
        land_ulog_t = ts_ld[i]
        break

if land_ulog_t is None:
    print('ERROR: No stable landed=true (>=0.5s) found after GOTO_ARRIVED')
    sys.exit(1)

# 插值 ENU 在接地时刻
land_e = float(np.interp(land_ulog_t, ts_lp, e_enu))
land_n = float(np.interp(land_ulog_t, ts_lp, n_enu))
land_u = float(np.interp(land_ulog_t, ts_lp, u_enu))

print(f'\nStable landing: ULog t={land_ulog_t:.3f}s, {cons}×landed lasting {dur:.2f}s')
print(f'  ENU: E={land_e:.3f}, N={land_n:.3f}, U={land_u:.3f}')

# ── 6. 提取轨迹切片 ─────────────────────────────────────────
mask = (ts_lp >= ts_lp[start_idx]) & (ts_lp <= land_ulog_t)
t_traj  = ts_lp[mask]
e_traj  = e_enu[mask]
n_traj  = n_enu[mask]
u_traj  = u_enu[mask]

# 重采样 (50Hz)
dt = 0.02
t_rs = np.arange(ts_lp[start_idx], land_ulog_t + dt, dt)
e_rs = np.interp(t_rs, t_traj, e_traj)
n_rs = np.interp(t_rs, t_traj, n_traj)
u_rs = np.interp(t_rs, t_traj, u_traj)

traj_dur = land_ulog_t - ts_lp[start_idx]
print(f'\nTrajectory: {len(t_rs)} pts @ 50Hz, {traj_dur:.1f}s real')
print(f'  Start ENU: E={e_traj[0]:.2f}, N={n_traj[0]:.2f}, U={u_traj[0]:.2f}')
print(f'  End ENU:   E={e_traj[-1]:.2f}, N={n_traj[-1]:.2f}, U={u_traj[-1]:.2f}')
print(f'  Height drop: {u_traj[0] - u_traj[-1]:.1f}m')

# ── 7. 侧视投影 ─────────────────────────────────────────────
# 水平方向: 安全点抵达位置 → 真实接地点
dx = land_e - e_traj[0]
dy = land_n - n_traj[0]
proj_len = np.sqrt(dx**2 + dy**2)
if proj_len < 1e-6:
    proj_len = 1.0
    dx, dy = 1.0, 0.0
ux, uy = dx / proj_len, dy / proj_len

# 投影所有点 (接地点为横坐标原点)
h_dist = np.zeros(len(t_rs))
for i in range(len(t_rs)):
    h_dist[i] = (e_rs[i] - land_e) * ux + (n_rs[i] - land_n) * uy

# 高度: 相对接地点
heights = u_rs - land_u

total_h = abs(h_dist[-1] - h_dist[0])
print(f'\nProjection: {proj_len:.2f}m safe→land horizontal')
print(f'  Safe arrival h_dist={h_dist[0]:.3f}m, height={heights[0]:.1f}m')
print(f'  Landing     h_dist={h_dist[-1]:.3f}m, height={heights[-1]:.3f}m')

# ── 8. 绘图 ──────────────────────────────────────────────────
dpi = 100
fig = plt.figure(figsize=(W/dpi, H/dpi), dpi=dpi,
                facecolor='none', edgecolor='none')
ax = fig.add_axes([0.08, 0.12, 0.88, 0.80], facecolor='none')

# 画布范围
h_margin = max(total_h * 0.1, 3)
v_margin = max(heights.max() * 0.08, 3)
h_min = h_dist.min() - h_margin
h_max = h_dist.max() + h_margin
v_min = -v_margin
v_max = heights.max() + v_margin

# 保持等比例
plot_w = W * 0.88
plot_h = H * 0.80
if (h_max - h_min) / (v_max - v_min) < plot_w / plot_h:
    need = (v_max - v_min) * plot_w / plot_h
    expand = (need - (h_max - h_min)) / 2
    h_min -= expand; h_max += expand
else:
    need = (h_max - h_min) / plot_w * plot_h
    expand = (need - (v_max - v_min)) / 2
    v_min -= expand; v_max += expand

ax.set_xlim(h_min, h_max)
ax.set_ylim(v_min, v_max)
ax.set_aspect('equal')
ax.axis('off')

# 地面基准线 (已移除)

# 安全点抵达标记 (绿色, 在轨迹首点)
ax.plot(h_dist[0], heights[0], 'o', color='#00FF88', markersize=14,
        markeredgecolor='white', markeredgewidth=2, alpha=0.95, zorder=30)
ax.annotate('Safe Point\nArrival', (h_dist[0], heights[0]),
            textcoords='offset points', xytext=(14, 14),
            color='#00FF88', fontsize=12, fontweight='bold', alpha=0.9,
            va='bottom')

# 接地点标记 (橙色)
ax.plot(h_dist[-1], heights[-1], 's', color='#FF8800', markersize=14,
        markeredgecolor='white', markeredgewidth=2, alpha=0.95, zorder=30)
ax.annotate('Landing', (h_dist[-1], heights[-1]),
            textcoords='offset points', xytext=(10, -22),
            color='#FF8800', fontsize=12, fontweight='bold', alpha=0.9,
            va='top')

# 动态元素
traj_line, = ax.plot([], [], '-', color='#00E5FF', alpha=0.9,
                      linewidth=3.5, solid_capstyle='round', zorder=20)
aircraft, = ax.plot([], [], 'o', color='white', markersize=9,
                     markeredgecolor='#00E5FF', markeredgewidth=2,
                     alpha=1.0, zorder=40)

# 文字 HUD
txt_title = ax.text(0.99, 0.98, 'Orin Landing · Descent Phase',
                    transform=ax.transAxes, fontsize=16, color='white',
                    alpha=0.9, fontweight='bold', ha='right', va='top',
                    fontfamily='sans-serif')
txt_info = ax.text(0.99, 0.92, '', transform=ax.transAxes,
                   fontsize=11, color='white', alpha=0.65,
                   fontfamily='monospace', ha='right', va='top',
                   linespacing=1.6)
txt_time = ax.text(0.01, 0.02, '', transform=ax.transAxes,
                   fontsize=13, color='white', alpha=0.75,
                   fontfamily='monospace', va='bottom')

# ── 9. 帧渲染 ────────────────────────────────────────────────
tmpdir = tempfile.mkdtemp(prefix='traj_side_')
print(f'\nRendering {tmpdir} ...')

PAUSE_START = 0.5
PAUSE_END   = 1.0
total_frames  = int(DUR * FPS)
pause_start_n = int(PAUSE_START * FPS)
pause_end_n   = int(PAUSE_END * FPS)
anim_n        = total_frames - pause_start_n - pause_end_n

# 真实飞行时长
FLIGHT_DUR = traj_dur

for fi in range(total_frames):
    if fi < pause_start_n:
        idx = 0
        t_disp = 0.0
    elif fi >= total_frames - pause_end_n:
        idx = len(t_rs) - 1
        t_disp = FLIGHT_DUR
    else:
        progress = (fi - pause_start_n) / anim_n
        t_disp = progress * FLIGHT_DUR
        t_real = ts_lp[start_idx] + t_disp
        idx = min(np.searchsorted(t_rs, t_real), len(t_rs) - 1)

    # 已飞行轨迹
    traj_line.set_data(h_dist[:idx+1], heights[:idx+1])

    # 航机标记
    h_cur = h_dist[idx]
    v_cur = heights[idx]
    aircraft.set_data([h_cur], [v_cur])

    # 高度指示虚线
    hline = ax.plot([h_cur, h_cur], [0, v_cur], ':',
                    color='#00E5FF', alpha=0.18, linewidth=1, zorder=2)

    # 文字
    m, s = divmod(t_disp, 60)
    txt_time.set_text(
        f'{int(m):01d}:{s:04.1f}  /  '
        f'{int(FLIGHT_DUR//60):01d}:{FLIGHT_DUR%60:04.1f}')

    rem_dist = abs(h_dist[-1] - h_cur)
    txt_info.set_text(
        f'Altitude       {v_cur:5.1f} m\n'
        f'To landing     {rem_dist:5.1f} m\n'
        f'H dist total   {total_h:5.1f} m'
    )

    fig.savefig(os.path.join(tmpdir, f'frame_{fi:05d}.png'),
                dpi=dpi, facecolor='none', edgecolor='none',
                transparent=True, pad_inches=0)

    for line in hline:
        line.remove()

    if (fi + 1) % 60 == 0:
        print(f'  {fi+1}/{total_frames} frames')

print(f'All {total_frames} frames rendered.')

# ── 10. FFmpeg 编码 ──────────────────────────────────────────
print(f'\nEncoding {OUT} ...')
cmd = (
    f'ffmpeg -y -framerate {FPS} '
    f'-i "{tmpdir}/frame_%05d.png" '
    f'-c:v prores_ks -profile:v 4444 -pix_fmt yuva444p10le '
    f'-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
    f'"{OUT}" 2>&1'
)
ret = os.system(cmd)
if ret != 0:
    print('ProRes failed, trying PNG codec...')
    ret = os.system(
        f'ffmpeg -y -framerate {FPS} '
        f'-i "{tmpdir}/frame_%05d.png" '
        f'-c:v png -pix_fmt rgba '
        f'"{OUT}" 2>&1'
    )
    if ret != 0:
        print(f'\n⚠️  FFmpeg failed. Frames: {tmpdir}')
        sys.exit(1)

shutil.rmtree(tmpdir, ignore_errors=True)

mb = os.path.getsize(OUT) / (1024 * 1024)
print(f'\n✅ {OUT}')
print(f'   {W}×{H} @ {FPS}fps, {DUR}s, ProRes 4444 + Alpha, {mb:.1f} MB')
print(f'   Start match error: {start_err:.3f}m')
print(f'   Descent: {traj_dur:.1f}s, {len(t_rs)} samples')
print(f'   Start→End ENU: ({e_traj[0]:.1f},{n_traj[0]:.1f},{u_traj[0]:.1f}) → ({e_traj[-1]:.1f},{n_traj[-1]:.1f},{u_traj[-1]:.1f})')
