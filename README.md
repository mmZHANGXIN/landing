# Orin Landing

> 基于 Jetson Orin NX 16GB + Mid360 LiDAR + FAST-LIO + HALSS + ONNX DRL (ROS1 Noetic 版)


## 依赖环境

| 组件 | 版本 | 说明 |
|------|------|------|
| 硬件 | Jetson Orin NX 16GB | ARM64, CUDA Cores + Tensor Cores |
| 系统 | Ubuntu 20.04 + JetPack 5.x | 自带 CUDA 11.4, cuDNN 8.5, TensorRT 8.5 |
| ROS | **Noetic** (ROS1) | FAST-LIO + livox_ros_driver2 运行环境 |
| Python | 3.8 (conda: fylanding) | `/home/ifsc_orin/miniconda3/envs/fylanding` |
| PyTorch | 2.1+ (Jetson版) | NVIDIA 官方 Jetson wheel |
| OpenCV | 4.5+ | 图像处理与可视化 |
| ONNX Runtime | 1.17+ | DRL 推理 (零 TensorFlow 依赖) |
| MAVSDK | 2.8+ | PX4 飞控通信 |

## 快速部署

### Step 1: 系统环境

```bash
# 确认系统架构 (需为 aarch64 / Jetson Orin)
uname -m
# 确认 CUDA 版本
nvcc --version
# ROS1 Noetic 必须已安装
source /opt/ros/noetic/setup.bash && echo "ROS Noetic OK"
```

### Step 2: 安装 Livox-SDK2 + livox_ros_driver2

```bash
# Livox-SDK2 (系统库)
cd ~
git clone https://github.com/Livox-SDK/Livox-SDK2.git
cd Livox-SDK2 && mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(nproc) && sudo make install

# livox_ros_driver2 (ROS1 版)
mkdir -p ~/livox_ws/src
cd ~/livox_ws/src
git clone https://github.com/Livox-SDK/livox_ros_driver2.git
cd livox_ros_driver2
source /opt/ros/noetic/setup.bash
./build.sh ROS1

# 修改 MID360 IP 配置
# 默认: host=192.168.1.5, lidar=192.168.1.103
vim ~/livox_ws/src/livox_ros_driver2/config/MID360_config.json
```

### Step 3: 编译 FAST-LIO

```bash
mkdir -p ~/fast_lio_ws/src
# 复制 arch/FAST_LIO 并适配 livox_ros_driver2
cp -r ~/evelyn/landing/arch/FAST_LIO ~/fast_lio_ws/src/
cd ~/fast_lio_ws/src/FAST_LIO
# 修改依赖: livox_ros_driver → livox_ros_driver2
sed -i 's/livox_ros_driver</livox_ros_driver2</g' package.xml
sed -i 's/livox_ros_driver/livox_ros_driver2/g' CMakeLists.txt
find . -name "*.h" -o -name "*.cpp" | xargs sed -i \
  's|livox_ros_driver/|livox_ros_driver2/|g; s|livox_ros_driver::|livox_ros_driver2::|g'
# 补充 ikd-Tree 子模块
git clone --depth 1 --branch fast_lio \
  https://github.com/hku-mars/ikd-Tree.git include/ikd-Tree

# 编译
cd ~/fast_lio_ws
source /opt/ros/noetic/setup.bash
source ~/livox_ws/devel/setup.bash
catkin_make
```

### Step 4: Python 环境

```bash
# 使用 conda 环境 fylanding (Python 3.8)
# 核心依赖已安装: numpy, opencv-python, pyyaml, matplotlib, onnxruntime, scipy
# ROS1 Python 支持: rospkg, pycryptodomex
PYTHON=/home/ifsc_orin/miniconda3/envs/fylanding/bin/python
$PYTHON -c "import rospy, numpy, cv2, yaml, onnxruntime; print('OK')"
```

### Step 5: 准备 ONNX 模型

```bash
# ONNX 模型需在 x86 训练机导出 (一次性):
# python scripts/export_ppo2_to_onnx.py \
#     --model arch/DeepRL/data/trained_policy/last_step_model.zip \
#     --output weights/ppo2_policy.onnx
ls weights/ppo2_policy.onnx  # 确认存在
```

### Step 6: 配置参数

编辑 `config/experiment_config.yaml`，关键参数:
- `perception.halss_weight_path` — HALSS Bayesian UNet 权重路径
- `uav.yaw_rate_rad_s` — 偏航转速 (rad/s)，可在总配置、场景 profile
  或命令行中设置；软件不限制其数值范围
- `uav.action_lateral_sign` — -1 对齐原 DeepRL，+1 镜像模式

### Step 7: 运行

#### 室内/室外定位与 QGC 检查

FAST-LIO 只维护一份 `config/mid360.yaml`。室内允许向 MAVROS 注入重力对齐位姿，
室外必须关闭该发布：

QGroundControl 参数由操作员配置，程序不会写入飞控。改完 EKF2 参数后重启飞控，
并在正式实验前导出参数文件到本次 `experiments/runs/<run>/` 目录。

| 场景 | PX4 1.13 | PX4 1.14+ |
|---|---|---|
| 室内位置 | `EKF2_AID_MASK` 启用 vision position | `EKF2_EV_CTRL` bit 0 启用水平位置；按需要启用 bit 1 垂直位置 |
| 室内 yaw | 仅在 FAST-LIO yaw 已验证连续且坐标一致后启用 vision yaw | 仅在验证后启用 `EKF2_EV_CTRL` bit 3 |
| 室内高度 | 按实验选择 `EKF2_HGT_MODE`，使用 vision 时选 Vision | 按实验选择 `EKF2_HGT_REF`，使用 vision 时选 Vision |
| 室外 | `EKF2_AID_MASK` 使用 GPS，关闭 vision position/yaw | 使用 GNSS aiding，`EKF2_EV_CTRL` 关闭 EV position/yaw |

室内还需在 QGC 检查 external-vision innovation、local position valid，并避免弱 GPS 参与水平融合；
室外需检查 GPS fix、home 和 local/global position，并用短时 `rostopic echo` 确认
`/mavros/vision_pose/pose` 没有实际消息。FAST-LIO关闭注入时仍可能注册publisher，
所以不能只根据 `rostopic info` 判断。参数含义以
[PX4 1.13 External Position Estimation](https://docs.px4.io/v1.13/en/ros/external_position_estimation)
和 [PX4 1.14 EKF2 External Vision](https://docs.px4.io/v1.14/en/advanced_config/tuning_the_ecl_ekf.html#external-vision-system)
为准，不在文档中硬编码未经现场验证的 bitmask 数值。

正式飞行默认启动 rosbag。录包失败会在等待解锁前终止；台架调试可显式使用
`--no-record-bag`。解锁、记录 home、垂直起飞和高空稳定阶段使用固定 yaw；
到达 Phase 1 集结点后，程序先原地保持位置并发送配置的 yaw-rate 旋转2秒，
随后进入 `GOTO_SAFE`，该 yaw-rate 继续贯穿 GOTO、warmup 与 DRL 降落。
MAVROS 后台线程始终以 `flight_controller.setpoint_rate_hz` 重发当前完整
position/velocity＋yaw 或 yaw-rate setpoint；推理只更新平移动作，不决定偏航心跳频率。

室外程序在人工起飞前锁定 PX4 ENU 地面原点。飞手人工升高并切入 OFFBOARD 后，
程序以水平不超过2m/s、垂直不超过1m/s的比例速度返回地面原点正上方30m；空中接管点
不会被重新定义为高度零点。随后 `GOTO_SAFE` 使用同样的 XYZ 限速，接近目标时按
`v=kp×误差` 减速。GOTO 超时后固定当前位置与航向，进入 `HOLD_FOR_MANUAL`；飞手切出
OFFBOARD 会被记录为正常 `MANUAL_TAKEOVER`，程序不会自动解锁或误触发急停。

室外感知不再使用 `/ali_odom` 控制或裁剪 `/ali_cloud`。主管线订阅
`/cloud_registered_body`，应用 Mid360 IMU→`base_link` 安装外参，并用同时间戳的 PX4
roll/pitch 构造重力水平机体系 ROI。PX4 EKF/GPS 是 FSM、位置、高度、姿态和动作旋转的
唯一来源；`/ali_odom`、`/ali_cloud` 只保留在 rosbag 中诊断。点云与 PX4 odometry 时间差
超过100ms、点云过期/稀疏或 PX4 姿态超过门限时，DRL 不执行并保持等待处理。

FAST-LIO 必须每次实验重新启动并在静止状态初始化。草坪等大平面会使水平平移和 yaw
弱可观，单天线 GPS 也不能提供可靠 yaw。若后续增加 GPS-LIO，应通过带时间戳、协方差、
杆臂与异常值门控的测量更新或因子图维护 `map_gps→odom_lio`，禁止直接覆盖 FAST-LIO pose。
软件不对 yaw-rate 设置数值门控；实际可执行范围仍受 PX4 参数、机体动力学和场地条件约束。
FAST-LIO 同时发布 `/fastlio/degeneracy_metrics`，数组依次为时间戳、有效约束数、平均残差、
pose 信息矩阵最小/最大特征值、条件数、LiDAR-IMU时间差和位置范数；这些量先用于录包标定
草坪/结构化场景阈值，本版本不依据未经标定的固定阈值修改滤波器状态。

每次运行在 `experiments/runs/<timestamp>_orin_landing/` 中固定生成：

- `mission_events.csv`：记录 `ARMED`、`TAKEOFF_STARTED`、
  `HIGH_ALTITUDE_REACHED`、`GOTO_STARTED`、`GOTO_ARRIVED`、
  `DRL_DESCENT_STARTED`、`DIRECT_LAND_STARTED`、`PX4_ON_GROUND` 和
  `DISARMED`。事件同时保存 ROS 时间、单调时间、MAVROS ENU、FAST-LIO 位姿和状态。
- `frame_timing.csv`：每个实际执行点云感知的坐标变换/ROI 预处理、HALSS、投影、
  深度补全、ONNX、控制和总耗时，以及点云 header 时间、结果年龄、是否接受。GPU warmup 和 `DIRECT_LAND`
  占位图不计入感知帧。
- `perception_gate_log.csv`：每个候选点云的 PX4 时间同步差、点数、有限点比例、ROI点数、
  PX4/FAST-LIO两套位姿、实际门控阶段和拒绝原因；可直接区分“未进入推理”和“推理超龄”。

其中 `perception_inference_ms = pointcloud_preprocess_ms + halss_ms +
depth_projection_ms + depth_completion_ms + onnx_ms`；统计脚本默认只对
`accepted=1`、即真正用于控制的推理结果求平均值。

实验后可直接统计阶段耗时及单帧感知平均值、P50、P95：

```bash
python3 scripts/summarize_run.py experiments/runs/<timestamp>_orin_landing
```

`HIGH_ALTITUDE_REACHED → PX4_ON_GROUND` 是高空原点到接地总时间；
`DRL_DESCENT_STARTED → PX4_ON_GROUND` 是 DRL 降落时间。接地事件只接受
`/mavros/extended_state` 的 `ON_GROUND`，不会用高度阈值冒充接地。

#### 7b. 无飞控测试 — FAST-LIO 去畸变路径

```bash
# 终端 1: roscore
roscore &

# 终端 2: MID360 CustomMsg 模式 (FAST-LIO 需要点级时间戳)
sudo ip addr add 192.168.1.5/24 dev eth0

source /opt/ros/noetic/setup.bash && source ~/livox_ws/devel/setup.bash
roslaunch livox_ros_driver2 msg_MID360.launch rviz_enable:=false

# 终端 3: FAST-LIO
source ~/fast_lio_ws/devel/setup.bash
roslaunch fast_lio mapping_mid360.launch rviz:=false
roslaunch fast_lio frontend_mid360.launch

## 室内：FAST-LIO → /mavros/vision_pose/pose → PX4 EKF2
roslaunch fast_lio mapping_mid360.launch external_vision:=true rviz:=false
python pipeline.py --config ./config/experiment_indoor_fastlio.yaml --mode ros

# 完整 SLAM launch 同时发布 /ali_odom、/ali_cloud、/cloud_registered_body、
# /fastlio/degeneracy_metrics 和 /mavros/vision_pose/pose。室内主管线使用
# /cloud_registered_body 做与室外一致的感知预处理，使用 /ali_odom 做定位健康检查。

## 室外：GPS → PX4 EKF2；FAST-LIO 仅供感知
roslaunch fast_lio mapping_mid360.launch external_vision:=false rviz:=false
python pipeline.py --config ./config/experiment_outdoor_gps.yaml --mode ros


# 终端 4: 验证 FAST-LIO 输出
source /opt/ros/noetic/setup.bash
rostopic hz /cloud_registered
rostopic hz /Odometry
rostopic echo /Odometry -n 1
rostopic echo /mavros/local_position/pose

roslaunch mavros px4.launch 
##
# pose.pose.position:
#   x: 0.12
#   y: -0.05
#   z: 0.00
# orientation:
#   x: 0.0, y: 0.0, z: 0.0, w: 1.0

## 实时性能

验收以 Orin + Mid360 + FAST-LIO 现场日志的 P50/P95/max 为准：

| 环节 | 实现 | 验收 |
|------|------|------|
| FAST-LIO | 外部 ROS1/C++ 节点输出 `/cloud_registered` 与 `/Odometry` | `rostopic hz` 确认频率 |
| HALSS Bayesian | CPU surface-normal + CUDA UNet/MC Dropout | 日志 `H=` P50/P95 |
| 深度投影/NN-fill | 几何最近邻 + 平滑 (raw) 或 PyTorch CUDA 透视 (FAST-LIO) | 日志 `D=` P50/P95 |
| ONNX DRL | ONNX Runtime 推理，零 TF 依赖 | 日志 `RL=` P50/P95 |
| **闭环总耗时** | 串行闭环 | P95 ≤100ms，超预算帧 ≤5% |
