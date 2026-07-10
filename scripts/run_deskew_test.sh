#!/bin/bash
# ============================================================
# run_deskew_test.sh — FAST-LIO 去畸变效果测试 (ROS2 Galactic)
# ============================================================
# 环境: ROS2 Galactic + ~/livox_ws (FAST-LIO)
#
# 用法:
#   离线回放模式:
#     ./scripts/run_deskew_test.sh bags/deskew_test/tilt_30deg
#
#   在线模式 (直接用 livox_ros_driver2):
#     ./scripts/run_deskew_test.sh --live
#
# 功能:
#   1. 启动 FAST-LIO (mid360 配置)
#   2. 启动 RViz2 (去畸变对比视图)
#   3. (可选) 回放 rosbag
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RVIZ_CFG="${PROJECT_ROOT}/arch/FAST_LIO/rviz_cfg/deskew_comparison.rviz"

# ---- ROS2 环境 ----
ROS2_SETUP="/opt/ros/galactic/setup.bash"
LIVOX_SETUP="${HOME}/livox_ws/install/setup.bash"

MODE="offline"
BAG_PATH=""

# ---- 解析参数 ----
if [ "$1" == "--live" ]; then
    MODE="live"
elif [ -n "$1" ]; then
    BAG_PATH="$1"
    if [ ! -d "${PROJECT_ROOT}/$BAG_PATH" ] && [ ! -d "$BAG_PATH" ]; then
        if [ -d "$1" ]; then
            BAG_PATH="$1"
        else
            echo "ERROR: Bag path not found: $1"
            echo "Usage: $0 [--live | <bag_path>]"
            exit 1
        fi
    else
        BAG_PATH="${PROJECT_ROOT}/$BAG_PATH"
    fi
else
    echo "Usage: $0 [--live | <bag_path>]"
    echo ""
    echo "Examples:"
    echo "  $0 --live                          # 在线模式"
    echo "  $0 bags/deskew_test/tilt_30deg     # 回放指定 rosbag"
    exit 1
fi

echo "============================================"
echo " FAST-LIO Deskew Test (ROS2 Galactic)"
echo "============================================"
echo " Mode:      ${MODE}"
if [ "${MODE}" == "offline" ]; then
    echo " Bag:       ${BAG_PATH}"
fi
echo " RViz cfg:  ${RVIZ_CFG}"
echo "============================================"

# ---- 加载 ROS2 环境 ----
if [ -f "${ROS2_SETUP}" ]; then
    source "${ROS2_SETUP}"
    echo "[OK] Sourced ROS2 Galactic"
else
    echo "ERROR: ROS2 Galactic not found at ${ROS2_SETUP}"
    exit 1
fi

if [ -f "${LIVOX_SETUP}" ]; then
    source "${LIVOX_SETUP}"
    echo "[OK] Sourced livox_ws (FAST-LIO)"
else
    echo "WARNING: livox_ws not found at ${LIVOX_SETUP}"
    echo "  FAST-LIO may not be available."
fi

# ---- 检查 RViz 配置 ----
if [ ! -f "${RVIZ_CFG}" ]; then
    echo "WARNING: Custom RViz config not found, using default."
    RVIZ_CFG="${PROJECT_ROOT}/arch/FAST_LIO/rviz_cfg/loam_livox.rviz"
fi

echo ""
echo " Starting FAST-LIO + RViz2..."
echo ""

# ============================================================
# 启动 FAST-LIO mapping 节点
# ============================================================
echo "[1/3] Launching FAST-LIO (mid360)..."
ros2 launch fast_lio mapping.launch.py rviz:=false &
LIO_PID=$!
sleep 4

# ============================================================
# 启动 RViz2 (去畸变对比视图)
# ============================================================
echo "[2/3] Launching RViz2..."
# 使用 ros2 run 避免 livox_ws 覆盖 rviz2 的 LD_LIBRARY_PATH
ros2 run rviz2 rviz2 -d "${RVIZ_CFG}" &
RVIZ_PID=$!
sleep 2

# ============================================================
# 回放 rosbag (离线模式)
# ============================================================
if [ "${MODE}" == "offline" ]; then
    echo "[3/3] Playing rosbag: ${BAG_PATH}"
    echo ""

    ros2 bag play "${BAG_PATH}" --clock

    echo ""
    echo " Bag playback finished."
    echo " Press Ctrl+C to stop FAST-LIO and RViz2."
fi

# ---- 等待用户 Ctrl+C ----
echo ""
echo "============================================"
echo " Deskew test running."
echo ""
echo " RViz2 Display Guide:"
echo "   RED    = /cloud_registered_body (body系, 去畸变后)"
echo "   GREEN  = /cloud_registered       (world系, 去畸变+位姿校正)"
echo "   ARROWS = /Odometry               (FAST-LIO 里程计)"
echo ""
echo " Key observations:"
echo "   - 倾斜雷达时, body系点云中地面是倾斜的"
echo "   - world系点云中地面应该恢复水平"
echo "   - 对比两者可判断去畸变+姿态估计效果"
echo "============================================"
echo " Press Ctrl+C to stop all."
echo ""

trap "kill ${LIO_PID} ${RVIZ_PID} 2>/dev/null; exit 0" INT TERM

# 保持运行
wait ${LIO_PID}
