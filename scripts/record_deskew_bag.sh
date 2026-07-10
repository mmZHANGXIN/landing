#!/bin/bash
# ============================================================
# record_deskew_bag.sh — 录制 MID360 去畸变测试 rosbag (ROS2 Galactic)
# ============================================================
# 环境: source /opt/ros/galactic/setup.bash
#
# 用法:
#   ./scripts/record_deskew_bag.sh [bag_name]
#
# 示例:
#   ./scripts/record_deskew_bag.sh tilt_30deg
#   # → 保存到 bags/deskew_test/tilt_30deg/
#
# 采集步骤:
#   1. 连接 MID360, 启动 livox_ros_driver2
#   2. 启动 FAST-LIO (另一个终端):
#        source /opt/ros/galactic/setup.bash && source ~/livox_ws/install/setup.bash
#        ros2 launch fast_lio mapping.launch.py rviz:=false
#   3. 手动倾斜雷达 (如 15°, 30°, 45°), 保持采集地面
#   4. 运行本脚本开始录制
#   5. 录制完成后 Ctrl+C 停止
# ============================================================

set -e

# ---- 加载 ROS2 环境 ----
ROS2_SETUP="/opt/ros/galactic/setup.bash"
if [ -f "${ROS2_SETUP}" ]; then
    source "${ROS2_SETUP}"
else
    echo "ERROR: ROS2 Galactic not found at ${ROS2_SETUP}"
    exit 1
fi

BAG_NAME="${1:-deskew_test}"
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)/bags/deskew_test"
OUTPUT_DIR="${BASE_DIR}/${BAG_NAME}"

# 如果文件夹已存在, 自动加后缀 _1, _2 ...
if [ -d "${OUTPUT_DIR}" ]; then
    i=1
    while [ -d "${OUTPUT_DIR}_${i}" ]; do
        i=$((i + 1))
    done
    echo "[WARN] '${OUTPUT_DIR}' already exists, using '${OUTPUT_DIR}_${i}' instead."
    OUTPUT_DIR="${OUTPUT_DIR}_${i}"
fi

echo "============================================"
echo " MID360 Deskew Test - Rosbag Recording"
echo "============================================"
echo " Output: ${OUTPUT_DIR}"
echo ""
echo " Topics to record:"
echo "   /livox/lidar            — 原始点云 (CustomMsg, FAST-LIO 输入)"
echo "   /livox/imu              — IMU 数据"
echo "   /cloud_registered_body  — body系去畸变点云 (FAST-LIO 输出)"
echo "   /cloud_registered       — world系去畸变点云 (FAST-LIO 输出)"
echo "   /Odometry               — 里程计位姿 (FAST-LIO 输出)"
echo "============================================"
echo ""
echo " Make sure livox_ros_driver2 AND FAST-LIO are running!"
echo " Press Ctrl+C to stop recording."
echo ""

ros2 bag record \
    -o "${OUTPUT_DIR}" \
    /livox/lidar \
    /livox/imu \
    /cloud_registered_body \
    /cloud_registered \
    /Odometry

echo ""
echo " Recording saved to: ${OUTPUT_DIR}"
echo " Done."
