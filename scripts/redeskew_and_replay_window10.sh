#!/usr/bin/env bash
# Re-run FAST-LIO deskewing from the raw Livox topics in a ROS1 bag, then
# optionally run replay_window10.py on the newly generated bag.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${APP_ROOT}/.." && pwd)"

RUN_NAME="20260807_162946_orin_landing"
POINT_FILTER_NUM=1
FASTLIO_MODE="frontend"
PLAY_RATE="1.0"
MODE="all"
FORCE="false"
NO_DISPLAY="false"
MAX_FRAMES=0
ROS_SETUP="/opt/ros/noetic/setup.bash"
FASTLIO_SETUP=""
PYTHON_BIN="${PYTHON_BIN:-python3}"
INPUT_BAG=""
OUTPUT_BAG=""
CONFIG_PATH=""
ONNX_MODEL="${APP_ROOT}/weights/ppo2_policy.onnx"
POSE_TOPIC="/mavros/local_position/odom"
FUSION_POSE_TOPIC=""
EXTRA_REPLAY_ARGS=()

usage() {
    cat <<'EOF'
用法:
  bash scripts/redeskew_and_replay_window10.sh [选项] [-- replay_window10额外参数]

默认行为:
  1. 从 input.bag 只回放 /livox/lidar、/livox/imu 和 MAVROS odom；
  2. 用 FAST-LIO frontend_only 模式和 point_filter_num=1 重新去畸变；
  3. 生成 input_redeskew_pf1.bag；
  4. 用新 bag 运行 replay_window10.py。

选项:
  --input BAG             原始 ROS1 bag
  --output BAG            新 bag（默认与 input.bag 同目录）
  --config YAML           实验配置快照
  --onnx-model FILE       PPO2 ONNX 模型
  --point-filter-num N    FAST-LIO 点过滤步长（默认 1）
  --full-lio              启用完整 ESKF、扫描匹配和地图更新，同时录制
                          /cloud_registered、/Odometry、/ali_cloud、/ali_odom
  --frontend-only         仅 IMU 前端去畸变（默认）
  --play-rate RATE        rosbag 回放倍率（默认 1.0；完整后端 PF=1 建议 0.5）
  --pose-topic TOPIC      回放使用的位姿（默认 /mavros/local_position/odom）
  --fusion-pose-topic T   window10 多帧融合位姿；完整 Fast-LIO 模式默认
                          /ali_odom，前端模式默认沿用 --pose-topic
  --fastlio-setup FILE    FAST-LIO catkin 工作空间的 devel/setup.bash
                          （README 默认：~/fast_lio_ws/devel/setup.bash）
  --ros-setup FILE        ROS1 setup.bash（默认 /opt/ros/noetic/setup.bash）
  --python FILE           运行离线 Python 管线的解释器
  --deskew-only           只生成新 bag
  --replay-only           不跑 FAST-LIO，直接回放 --output 指定的新 bag
  --no-display            replay_window10 关闭图形窗口
  --max-frames N          replay_window10 最多处理 N 帧
  --force                 输出 bag 已存在时先改名备份，再重新生成
  -h, --help              显示帮助

示例:
  bash scripts/redeskew_and_replay_window10.sh --deskew-only
  bash scripts/redeskew_and_replay_window10.sh --full-lio --deskew-only
  bash scripts/redeskew_and_replay_window10.sh --replay-only --no-display
  bash scripts/redeskew_and_replay_window10.sh -- --save-dir /tmp/window10_pf1
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input) INPUT_BAG="$2"; shift 2 ;;
        --output) OUTPUT_BAG="$2"; shift 2 ;;
        --config) CONFIG_PATH="$2"; shift 2 ;;
        --onnx-model) ONNX_MODEL="$2"; shift 2 ;;
        --point-filter-num) POINT_FILTER_NUM="$2"; shift 2 ;;
        --full-lio) FASTLIO_MODE="full"; shift ;;
        --frontend-only) FASTLIO_MODE="frontend"; shift ;;
        --play-rate) PLAY_RATE="$2"; shift 2 ;;
        --pose-topic) POSE_TOPIC="$2"; shift 2 ;;
        --fusion-pose-topic) FUSION_POSE_TOPIC="$2"; shift 2 ;;
        --fastlio-setup) FASTLIO_SETUP="$2"; shift 2 ;;
        --ros-setup) ROS_SETUP="$2"; shift 2 ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        --deskew-only) MODE="deskew"; shift ;;
        --replay-only) MODE="replay"; shift ;;
        --no-display) NO_DISPLAY="true"; shift ;;
        --max-frames) MAX_FRAMES="$2"; shift 2 ;;
        --force) FORCE="true"; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; EXTRA_REPLAY_ARGS=("$@"); break ;;
        *) echo "ERROR: 未知参数: $1" >&2; usage; exit 2 ;;
    esac
done

if ! [[ "${POINT_FILTER_NUM}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --point-filter-num 必须是正整数" >&2
    exit 2
fi
if ! awk -v rate="${PLAY_RATE}" 'BEGIN {exit !(rate ~ /^[0-9]+([.][0-9]+)?$/ && rate > 0)}'; then
    echo "ERROR: --play-rate 必须是正数" >&2
    exit 2
fi

find_run_dir() {
    local candidate
    for candidate in \
        "${WORKSPACE_ROOT}/experiments/${RUN_NAME}" \
        "${WORKSPACE_ROOT}/experiments/runs/${RUN_NAME}" \
        "${APP_ROOT}/experiments/${RUN_NAME}" \
        "${APP_ROOT}/experiments/runs/${RUN_NAME}"; do
        if [[ -d "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    return 1
}

RUN_DIR="$(find_run_dir || true)"
if [[ -z "${RUN_DIR}" && ( -z "${INPUT_BAG}" || -z "${CONFIG_PATH}" ) ]]; then
    echo "ERROR: 找不到实验目录 ${RUN_NAME}，请显式传 --input 和 --config" >&2
    exit 1
fi

INPUT_BAG="${INPUT_BAG:-${RUN_DIR}/input.bag}"
if [[ "${FASTLIO_MODE}" == "full" ]]; then
    OUTPUT_BAG="${OUTPUT_BAG:-${RUN_DIR}/input_fastlio_full_pf${POINT_FILTER_NUM}.bag}"
else
    OUTPUT_BAG="${OUTPUT_BAG:-${RUN_DIR}/input_redeskew_pf${POINT_FILTER_NUM}.bag}"
fi
CONFIG_PATH="${CONFIG_PATH:-${RUN_DIR}/experiment_config_snapshot.yaml}"
if [[ -z "${FUSION_POSE_TOPIC}" ]]; then
    if [[ "${FASTLIO_MODE}" == "full" ]]; then
        FUSION_POSE_TOPIC="/ali_odom"
    else
        FUSION_POSE_TOPIC="${POSE_TOPIC}"
    fi
fi

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "ERROR: 文件不存在: $1" >&2
        exit 1
    fi
}

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "ERROR: 找不到命令: $1" >&2
        exit 1
    fi
}

source_ros_environment() {
    require_file "${ROS_SETUP}"
    # ROS setup scripts are not nounset-safe, so this script deliberately uses
    # set -e/pipefail without set -u.
    source "${ROS_SETUP}"

    if [[ -n "${FASTLIO_SETUP}" ]]; then
        require_file "${FASTLIO_SETUP}"
        source "${FASTLIO_SETUP}"
        if ! rospack find fast_lio >/dev/null 2>&1; then
            echo "[WARN] ${FASTLIO_SETUP} 已加载，但其中没有 fast_lio；尝试 README 中的 ~/fast_lio_ws" >&2
            FASTLIO_SETUP=""
        fi
    fi

    # README.md 的 Orin 布局：Livox driver 在 ~/livox_ws，FAST-LIO 在
    # ~/fast_lio_ws。driver setup 先加载，避免 CustomMsg 运行时依赖缺失。
    if [[ -f "${HOME}/livox_ws/devel/setup.bash" ]]; then
        source "${HOME}/livox_ws/devel/setup.bash"
    fi
    if ! rospack find fast_lio >/dev/null 2>&1; then
        local candidate
        for candidate in \
            "${HOME}/fast_lio_ws/devel/setup.bash" \
            "${HOME}/catkin_ws/devel/setup.bash" \
            "${HOME}/livox_noetic_ws/devel/setup.bash"; do
            if [[ -f "${candidate}" ]]; then
                source "${candidate}"
                if rospack find fast_lio >/dev/null 2>&1; then
                    FASTLIO_SETUP="${candidate}"
                    break
                fi
            fi
        done
    fi
}

ROSCORE_PID=""
FASTLIO_PID=""
RECORDER_PID=""

stop_process() {
    local pid="$1"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
        kill -INT "${pid}" 2>/dev/null || true
        wait "${pid}" 2>/dev/null || true
    fi
}

cleanup() {
    stop_process "${RECORDER_PID}"
    stop_process "${FASTLIO_PID}"
    stop_process "${ROSCORE_PID}"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

start_master_if_needed() {
    if rostopic list >/dev/null 2>&1; then
        echo "[OK] 使用当前 ROS master"
        return
    fi
    local log_dir="${RUN_DIR:-$(dirname "${OUTPUT_BAG}")}/redeskew_logs"
    mkdir -p "${log_dir}"
    roscore >"${log_dir}/roscore.log" 2>&1 &
    ROSCORE_PID=$!
    local attempt
    for attempt in {1..50}; do
        if rostopic list >/dev/null 2>&1; then
            echo "[OK] 已启动临时 roscore (pid=${ROSCORE_PID})"
            return
        fi
        sleep 0.1
    done
    echo "ERROR: roscore 启动失败，见 ${log_dir}/roscore.log" >&2
    exit 1
}

bag_has_topic() {
    # Do not use grep -q here: with pipefail it can close the pipe early and
    # make rosbag's producer side report a false failure via SIGPIPE.
    rosbag info "$1" | grep -F "$2" >/dev/null
}

run_redeskew() {
    require_file "${INPUT_BAG}"
    source_ros_environment
    require_cmd rosbag
    require_cmd rosrun

    if ! rospack find fast_lio >/dev/null 2>&1; then
        echo "ERROR: ROS 环境中找不到 fast_lio；README 默认路径是 ~/fast_lio_ws/devel/setup.bash" >&2
        exit 1
    fi
    if ! bag_has_topic "${INPUT_BAG}" "/livox/lidar" || \
       ! bag_has_topic "${INPUT_BAG}" "/livox/imu" || \
       ! bag_has_topic "${INPUT_BAG}" "${POSE_TOPIC}"; then
        echo "ERROR: 原 bag 必须包含 /livox/lidar、/livox/imu 和 ${POSE_TOPIC}" >&2
        rosbag info "${INPUT_BAG}"
        exit 1
    fi

    if [[ -e "${OUTPUT_BAG}" || -e "${OUTPUT_BAG}.active" ]]; then
        if [[ "${FORCE}" != "true" ]]; then
            echo "ERROR: 输出已存在: ${OUTPUT_BAG}（用 --force 备份后重做）" >&2
            exit 1
        fi
        local backup_suffix
        backup_suffix="$(date +%Y%m%d_%H%M%S)"
        [[ -e "${OUTPUT_BAG}" ]] && mv "${OUTPUT_BAG}" "${OUTPUT_BAG}.bak_${backup_suffix}"
        [[ -e "${OUTPUT_BAG}.active" ]] && mv "${OUTPUT_BAG}.active" "${OUTPUT_BAG}.active.bak_${backup_suffix}"
    fi
    mkdir -p "$(dirname "${OUTPUT_BAG}")"

    start_master_if_needed
    rosparam set /use_sim_time true

    local fastlio_pkg config_file log_dir
    fastlio_pkg="$(rospack find fast_lio)"
    config_file="${fastlio_pkg}/config/mid360.yaml"
    require_file "${config_file}"
    log_dir="${RUN_DIR:-$(dirname "${OUTPUT_BAG}")}/redeskew_logs"
    mkdir -p "${log_dir}"

    # Load the normal MID360 geometry/extrinsics, then force the settings that
    # matter for deterministic offline deskew. Running rosrun directly avoids a
    # launch file silently overriding point_filter_num back to 3.
    rosparam load "${config_file}"
    if [[ "${FASTLIO_MODE}" == "full" ]]; then
        rosparam set /mapping/frontend_only_en false
    else
        rosparam set /mapping/frontend_only_en true
    fi
    rosparam set /publish/scan_publish_en true
    rosparam set /publish/scan_bodyframe_pub_en true
    rosparam set /publish/dense_publish_en true
    rosparam set /publish/path_en false
    rosparam set /publish/mavros_vision_pose_en false
    rosparam set /pcd_save/pcd_save_en false
    rosparam set /point_filter_num "${POINT_FILTER_NUM}"
    rosparam set /max_iteration 3
    rosparam set /filter_size_surf 0.5
    rosparam set /filter_size_map 0.5
    rosparam set /cube_side_length 1000
    rosparam set /runtime_pos_log_enable false

    rosrun fast_lio fastlio_mapping __name:=fastlio_redeskew \
        >"${log_dir}/fastlio_pf${POINT_FILTER_NUM}.log" 2>&1 &
    FASTLIO_PID=$!
    sleep 2
    if ! kill -0 "${FASTLIO_PID}" 2>/dev/null; then
        echo "ERROR: FAST-LIO 启动失败，见 ${log_dir}/fastlio_pf${POINT_FILTER_NUM}.log" >&2
        exit 1
    fi
    local actual_filter
    actual_filter="$(rosparam get /point_filter_num)"
    if [[ "${actual_filter}" != "${POINT_FILTER_NUM}" ]]; then
        echo "ERROR: /point_filter_num=${actual_filter}，预期 ${POINT_FILTER_NUM}" >&2
        exit 1
    fi
    local actual_frontend
    actual_frontend="$(rosparam get /mapping/frontend_only_en)"
    echo "[OK] FAST-LIO mode=${FASTLIO_MODE} frontend_only=${actual_frontend} point_filter_num=${actual_filter}"

    local record_topics=(
        /cloud_registered_body "${POSE_TOPIC}" /livox/lidar /livox/imu
    )
    if [[ "${FASTLIO_MODE}" == "full" ]]; then
        record_topics+=(/cloud_registered /Odometry /ali_cloud /ali_odom
                       /fastlio/degeneracy_metrics)
    fi
    rosbag record --buffsize=2048 -O "${OUTPUT_BAG}" "${record_topics[@]}" \
        >"${log_dir}/record_pf${POINT_FILTER_NUM}.log" 2>&1 &
    RECORDER_PID=$!
    sleep 2
    if ! kill -0 "${RECORDER_PID}" 2>/dev/null; then
        echo "ERROR: rosbag record 启动失败，见 ${log_dir}/record_pf${POINT_FILTER_NUM}.log" >&2
        exit 1
    fi

    echo "[RUN] 回放原始 LiDAR/IMU；旧 /cloud_registered_body 不会被发布"
    rosbag play --clock --rate "${PLAY_RATE}" "${INPUT_BAG}" --topics \
        /livox/lidar /livox/imu "${POSE_TOPIC}"
    sleep 2
    stop_process "${RECORDER_PID}"
    RECORDER_PID=""
    stop_process "${FASTLIO_PID}"
    FASTLIO_PID=""

    require_file "${OUTPUT_BAG}"
    if ! bag_has_topic "${OUTPUT_BAG}" "/cloud_registered_body"; then
        echo "ERROR: 新 bag 没有 /cloud_registered_body；见 ${log_dir}/fastlio_pf${POINT_FILTER_NUM}.log" >&2
        exit 1
    fi
    if [[ "${FASTLIO_MODE}" == "full" ]] && \
       ! bag_has_topic "${OUTPUT_BAG}" "/cloud_registered"; then
        echo "ERROR: 完整 Fast-LIO bag 没有 /cloud_registered；见 ${log_dir}/fastlio_pf${POINT_FILTER_NUM}.log" >&2
        exit 1
    fi
    echo "[OK] 新去畸变 bag: ${OUTPUT_BAG}"
    rosbag info "${OUTPUT_BAG}" | grep -E "duration:|/cloud_registered|/Odometry|/ali_cloud|/ali_odom|${POSE_TOPIC}|/livox/lidar|/livox/imu" || true
}

run_window10() {
    require_file "${OUTPUT_BAG}"
    require_file "${CONFIG_PATH}"
    require_file "${ONNX_MODEL}"
    if [[ -f "${ROS_SETUP}" ]]; then
        source "${ROS_SETUP}"
    fi
    if [[ -n "${FASTLIO_SETUP}" && -f "${FASTLIO_SETUP}" ]]; then
        source "${FASTLIO_SETUP}"
    fi
    require_cmd "${PYTHON_BIN}"

    local replay_args=(
        --bag "${OUTPUT_BAG}"
        --config "${CONFIG_PATH}"
        --cloud-source fastlio
        --cloud-topic /cloud_registered_body
        --pose-topic "${FUSION_POSE_TOPIC}"
        --raw-topic /livox/lidar
        --onnx-model "${ONNX_MODEL}"
        --semantic geometry
        --map-frame-mode world-first
        --window-size 30
        --bev-grid-res 64
    )
    [[ "${NO_DISPLAY}" == "true" ]] && replay_args+=(--no-display)
    [[ "${MAX_FRAMES}" -gt 0 ]] && replay_args+=(--max-frames "${MAX_FRAMES}")
    replay_args+=("${EXTRA_REPLAY_ARGS[@]}")

    echo "[RUN] replay_window10.py cloud=/cloud_registered_body fusion_pose=${FUSION_POSE_TOPIC}"
    cd "${APP_ROOT}"
    "${PYTHON_BIN}" scripts/replay_window10.py "${replay_args[@]}"
}

case "${MODE}" in
    all) run_redeskew; run_window10 ;;
    deskew) run_redeskew ;;
    replay) run_window10 ;;
esac
