#!/bin/bash
# ============================================================
# replay_orin_landing_bag.sh — 原始 Livox 点云回放 + 3D 可视化 (ROS1 Noetic)
# ============================================================
# 环境: source /opt/ros/noetic/setup.bash + 已运行的 roscore
#
# 用法:
#   ./scripts/replay_orin_landing_bag.sh [options]
#   conda deactivate
    # source /opt/ros/noetic/setup.bash
    # source ~/livox_noetic_ws/devel/setup.bash
    # python3 -c "from livox_ros_driver2.msg import CustomMsg; print('CustomMsg OK')"   # 验证
    # cd ~/evelyn/landing/scripts
    # bash replay_orin_landing_bag.sh
# 功能:
#   1. rosbag play --clock 只发布选定的话题, 不播 bag 中其余录制话题
#      (如 /mavros/*):
#        - 默认 (原始点云):  /livox/lidar + /livox/imu
#        - --deskew (去畸变): /cloud_registered_body + /livox/imu
#   2. 同时启动 replay_raw_livox_visualization.py, 实时订阅点云话题,
#      打开与 replay_bag_offline.py 一致风格的 Matplotlib 3D 点云窗口
#      (原始: 标题 4.Raw Livox Point Cloud / raw Livox 坐标系;
#       去畸变: 标题 4.Deskewed Point Cloud / level-body 坐标系)
#   3. 不运行 HALSS / 深度投影 / FAST-LIO, 只复现点云窗口,
#      不创建语义图和深度图窗口
#
# 参数:
#   --bag <path>    覆盖默认 bag (相对路径基于项目根)
#   --rate <倍率>   回放速率 (默认 1=实时, 0=尽可能快)
#   --loop          循环播放
#   --pause         启动后暂停 (等待按 s 继续)
#   --start <秒>    从 bag 起始偏移处开始
#   --deskew        播放去畸变点云 /cloud_registered_body (PointCloud2,
#                   FAST-LIO 录制输出) 而非原始 /livox/lidar
#   --no-display    仅播放, 不显示 3D 窗口
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_BAG="${PROJECT_ROOT}/experiments/20260807_162946_orin_landing/input.bag"

BAG_PATH=""
RATE=""
LOOP=""
PAUSE=""
START=""
DESKEW=""
NO_DISPLAY=""

usage() {
    cat <<'EOF'
用法: ./scripts/replay_orin_landing_bag.sh [options]

  --bag <path>    覆盖默认 bag (相对路径基于项目根)
  --rate <倍率>   回放速率 (默认 1=实时, 0=尽可能快)
  --loop          循环播放
  --pause         启动后暂停 (等待按 s 继续)
  --start <秒>    从 bag 起始偏移处开始
  --deskew        播放去畸变点云 /cloud_registered_body (PointCloud2)
                  而非原始 /livox/lidar (CustomMsg)
  --no-display    仅播放, 不显示 3D 窗口
EOF
}

# ---- 解析参数 ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bag)          BAG_PATH="$2"; shift 2 ;;
        --bag=*)        BAG_PATH="${1#*=}"; shift ;;
        --rate)         RATE="$2"; shift 2 ;;
        --rate=*)       RATE="${1#*=}"; shift ;;
        --loop)         LOOP="true"; shift ;;
        --pause)        PAUSE="true"; shift ;;
        --start)        START="$2"; shift 2 ;;
        --start=*)      START="${1#*=}"; shift ;;
        --deskew)       DESKEW="true"; shift ;;
        --no-display)   NO_DISPLAY="true"; shift ;;
        -h|--help)      usage; exit 0 ;;
        *) echo "Unknown arg: $1"; usage; exit 1 ;;
    esac
done

BAG_PATH="${BAG_PATH:-$DEFAULT_BAG}"
if [[ "${BAG_PATH}" != /* ]]; then
    BAG_PATH="${PROJECT_ROOT}/${BAG_PATH}"
fi

# ---- 检查输入文件 (Orin 数据在 experiments/runs/<name>/, 开发机在 experiments/<name>/) ----
# 指定路径不存在时自动尝试 runs/ 变体, 两种机器布局均直接可运行 (与 replay_bag_offline.py 一致)
if [ ! -f "${BAG_PATH}" ]; then
    BAG_DIR="$(dirname "${BAG_PATH}")"
    BAG_PARENT="$(dirname "${BAG_DIR}")"
    if [ "$(basename "${BAG_PARENT}")" == "experiments" ]; then
        ALT_BAG="${BAG_PARENT}/runs/$(basename "${BAG_DIR}")/$(basename "${BAG_PATH}")"
        if [ -f "${ALT_BAG}" ]; then
            echo "WARNING: ${BAG_PATH} 不存在, 改用 ${ALT_BAG}"
            BAG_PATH="${ALT_BAG}"
        fi
    fi
fi

if [ ! -f "${BAG_PATH}" ]; then
    echo "ERROR: rosbag not found: ${BAG_PATH}"
    exit 1
fi

# ---- 加载 ROS1 Noetic 环境 ----
ROS1_SETUP="/opt/ros/noetic/setup.bash"
if [ -f "${ROS1_SETUP}" ]; then
    source "${ROS1_SETUP}"
else
    echo "ERROR: ROS1 Noetic not found at ${ROS1_SETUP}"
    exit 1
fi

# ---- 加载 livox_ros_driver2 的 ROS1 消息包 (若已编译) ----
# 需要 CustomMsg 消息类, 订阅 /livox/lidar 才能注册; 未安装时给出安装提示
LIVOX_NOETIC_SETUP="${HOME}/livox_noetic_ws/devel/setup.bash"
if [ -f "${LIVOX_NOETIC_SETUP}" ]; then
    source "${LIVOX_NOETIC_SETUP}"
    echo "[OK] Sourced ${LIVOX_NOETIC_SETUP}"
else
    echo "WARNING: ${LIVOX_NOETIC_SETUP} 不存在 — livox_ros_driver2 消息包未安装."
    echo "  订阅 /livox/lidar (CustomMsg) 需要它; 安装步骤见"
    echo "  replay_raw_livox_visualization.py 头部说明 (最小消息包)."
fi

# ---- 检查 roscore ----
if ! timeout 5 rostopic list >/dev/null 2>&1; then
    echo "ERROR: roscore is not running (ROS master unreachable)."
    echo "  Start it first:  roscore"
    exit 1
fi

# ---- 按模式选择播放话题 ----
if [ "${DESKEW}" == "true" ]; then
    CLOUD_TOPIC="/cloud_registered_body"
    MSG_TYPE_ARG="pointcloud2"
    MODE_LABEL="Deskewed Point Cloud (PointCloud2)"
else
    CLOUD_TOPIC="/livox/lidar"
    MSG_TYPE_ARG="custom"
    MODE_LABEL="Raw Livox Point Cloud (CustomMsg)"
fi
TOPICS=( "${CLOUD_TOPIC}" "/livox/imu" )

# ---- 检查 bag 话题 ----
BAG_INFO="$(rosbag info "${BAG_PATH}")"
if ! echo "${BAG_INFO}" | grep -q "${CLOUD_TOPIC}"; then
    echo "ERROR: bag has no ${CLOUD_TOPIC} topic: ${BAG_PATH}"
    echo "${BAG_INFO}"
    exit 1
fi
if ! echo "${BAG_INFO}" | grep -q "/livox/imu"; then
    echo "WARNING: bag has no /livox/imu topic (IMU will not be played)."
fi

echo "============================================"
echo " Orin Landing — Livox Replay + 3D Vis"
echo "============================================"
echo " Mode:      ${MODE_LABEL}"
echo " Bag:       ${BAG_PATH}"
[ -n "${RATE}" ] && echo " Rate:      ${RATE}x"
[ "${LOOP}" == "true" ] && echo " Loop:      on"
[ "${PAUSE}" == "true" ] && echo " Pause:     on"
[ -n "${START}" ] && echo " Start:     ${START}s"
[ "${NO_DISPLAY}" == "true" ] && echo " Display:   off"
echo " Topics:    ${CLOUD_TOPIC} /livox/imu"
echo "============================================"

# ---- 选择 Python 解释器 ----
# 优先 noetic 系统 python3 (/usr/bin/python3): 自带 rospy + 系统 matplotlib
# + python3-tk, TkAgg 窗口可正常弹出. conda 的 python3 虽有 rospy (pip),
# 但常缺 tkinter, matplotlib 会静默回退 Agg 后端导致无窗口 — 故必须排后.
PYTHON_BIN=""
for cand in /usr/bin/python3 python3; do
    if command -v "${cand}" >/dev/null 2>&1 && "${cand}" -c "import rospy" >/dev/null 2>&1; then
        PYTHON_BIN="${cand}"
        break
    fi
done
if [ -z "${PYTHON_BIN}" ]; then
    echo "ERROR: 未找到带 rospy 的 python3 (/usr/bin/python3 或 python3)"
    echo "  conda deactivate 后重试, 或安装 python3-roslib."
    exit 1
fi
echo "[OK] Python: ${PYTHON_BIN}"

PY_ARGS=(--msg-type "${MSG_TYPE_ARG}")
[ "${NO_DISPLAY}" == "true" ] && PY_ARGS+=(--no-display)
PY_SCRIPT="${SCRIPT_DIR}/replay_raw_livox_visualization.py"

echo ""
echo "[1/2] Starting 3D point cloud visualizer..."
"${PYTHON_BIN}" "${PY_SCRIPT}" "${PY_ARGS[@]}" &
PY_PID=$!

# 等待节点注册 (与项目其他启动脚本一致的 sleep 模式)
sleep 2
if ! kill -0 "${PY_PID}" 2>/dev/null; then
    echo "ERROR: visualizer exited early — check Python dependencies."
    echo "  Required: python3-rosbag / rospy + livox_ros_driver2 (或 livox_ros_driver) 消息包"
    exit 1
fi

# ---- 清理函数: 无论退出路径都关闭可视化进程 ----
cleanup() {
    if [ -n "${PY_PID}" ]; then
        kill "${PY_PID}" 2>/dev/null || true
        wait "${PY_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# ---- 回放 (只发布 /livox/lidar 与 /livox/imu) ----
PLAY_ARGS=(--clock)
[ -n "${RATE}" ] && PLAY_ARGS+=(--rate "${RATE}")
[ "${LOOP}" == "true" ] && PLAY_ARGS+=(--loop)
[ "${PAUSE}" == "true" ] && PLAY_ARGS+=(--pause)
[ -n "${START}" ] && PLAY_ARGS+=(--start "${START}")

echo ""
echo "[2/2] Playing bag (topics: ${CLOUD_TOPIC} /livox/imu)..."
echo "      Press Ctrl+C to stop."
echo ""

# rosbag play 语法随版本不同:
#   - pip 版 (pyrosbag, conda 环境常见): --bags=<bag> --topics /t1 /t2 (--topics 为布尔开关)
#   - 官方 Noetic 版:                    --topics="/t1 /t2" <bag>  (--topics 带值)
if rosbag play --help 2>&1 | grep -q -- "--bags"; then
    echo "[Info] rosbag: pip/pyrosbag syntax"
    rosbag play "${PLAY_ARGS[@]}" --bags="${BAG_PATH}" \
        --topics "${TOPICS[@]}"
else
    echo "[Info] rosbag: ROS1 Noetic syntax"
    rosbag play "${PLAY_ARGS[@]}" --topics="${TOPICS[*]}" "${BAG_PATH}"
fi

echo ""
echo " Playback finished. Closing visualizer..."
