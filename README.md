# Orin Landing - 无人机偏航故障着陆系统

> 基于 Jetson Orin NX 16GB + Mid360 LiDAR + FAST-LIO + HALSS + ONNX DRL (ROS1 Noetic 版)

## 架构概览

```
┌─────────────────────────────────────────────────────────────────────┐
│                    离线预处理（飞行前一次性）                          │
│  GIS卫星图 → SegFormer分割 → 全局安全先验栅格图                      │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    在线管线（飞行中实时循环，验收以实测 P95 为准）       │
│                                                                     │
│  Mid360 LiDAR ─→ FAST-LIO ──→ 去畸变点云 ─→ HALSS Bayesian ─→ 语义图   │
│                     ↑ IMU         │               安全评估            │
│                                    ├──────────→ 深度投影 ──→ 深度图    │
│                                                                     │
│  语义图(128²) + 深度图(128²) ──→ ONNX DRL ──→ 离散动作(0~9)           │
│                                                        ↓             │
│                                                NED速度 → MAVSDK → PX4 │
└─────────────────────────────────────────────────────────────────────┘

实时可视化 (4窗口):
  ┌──────────────┬──────────────┐
  │  语义图       │  深度图       │
  │  (安全=绿)    │  (热力图)     │
  ├──────────────┼──────────────┤
  │  BEV安全评估  │  飞行轨迹      │
  │  (俯视红绿)   │  (2D俯视)     │
  └──────────────┴──────────────┘
```

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
- `uav.yaw_rate_rad_s` — 偏航转速 (rad/s)
- `uav.action_lateral_sign` — -1 对齐原 DeepRL，+1 镜像模式

### Step 7: 运行

#### 7a. 无飞控测试 — raw 原始点云路径 (不依赖 FAST-LIO)

```bash
# 终端 1: roscore
roscore &

# 终端 2: MID360 PointCloud2 模式
sudo ip addr add 192.168.1.5/24 dev eth0
source /opt/ros/noetic/setup.bash && source ~/livox_ws/devel/setup.bash
roslaunch livox_ros_driver2 rviz_MID360.launch rviz_enable:=false

# 终端 3: 验证话题
source /opt/ros/noetic/setup.bash
rostopic hz /livox/lidar
rostopic hz /livox/imu

# 终端 4: 启动 landing
source /opt/ros/noetic/setup.bash
source ~/livox_ws/devel/setup.bash
/home/ifsc_orin/miniconda3/envs/fylanding/bin/python test_live_nocontrol_raw.py \
    --no-display --onnx-model weights/ppo2_policy.onnx
```

#### 7b. 无飞控测试 — FAST-LIO 去畸变路径

```bash
# 终端 1: roscore
roscore &

# 终端 2: MID360 CustomMsg 模式 (FAST-LIO 需要点级时间戳)
sudo ip addr add 192.168.1.5/24 dev eth0

source /opt/ros/noetic/setup.bash && source ~/livox_ws/devel/setup.bash
roslaunch livox_ros_driver2 msg_MID360.launch rviz_enable:=false

# 终端 3: FAST-LIO
source /opt/ros/noetic/setup.bash
source ~/livox_ws/devel/setup.bash
source ~/fast_lio_ws/devel/setup.bash
roslaunch fast_lio mapping_mid360.launch rviz:=false

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

# 终端 5: 录制 rosbag (可选)
source /opt/ros/noetic/setup.bash
rosbag record -O mid360_test.bag \
  /Odometry /cloud_registered /cloud_registered_body /livox/lidar /livox/imu

# 终端 6: 启动 landing
source /opt/ros/noetic/setup.bash
source ~/livox_ws/devel/setup.bash
source ~/fast_lio_ws/devel/setup.bash
/home/ifsc_orin/miniconda3/envs/fylanding/bin/python test_live_nocontrol.py --no-display
#### 7c. 真机闭环 (需 MAVSDK + PX4)

```bash
# 终端 1: MAVSDK 服务器
mavsdk_server /dev/ttyACM0 -p 14540

# 终端 2: 着陆管线
source /opt/ros/noetic/setup.bash
source ~/livox_ws/devel/setup.bash
source ~/fast_lio_ws/devel/setup.bash
/home/ifsc_orin/miniconda3/envs/fylanding/bin/python pipeline.py \
    --config ./config/experiment_config.yaml --mode ros \
    2>&1 | tee experiments/logs/pipeline.log
```

## 模块说明

### perception/
| 文件 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `halss_bayesian.py` | HALSS Bayesian 安全评估：CPU surface-normal 预处理 + CUDA UNet/MC Dropout 推理 | 点云 (N,3) | 安全概率, BEV网格 |
| `depth_projection.py` | 点云投影深度图 | 点云 + 位姿 | 稠密深度图 (128²) |
| `semantic_generator.py` | BEV → 语义图 | HALSS结果 | 语义类别图 (128²) |

### odometry/
| 文件 | 功能 |
|------|------|
| `fastlio_interface.py` | 订阅 FAST-LIO 的 ROS1 话题，获取位姿和去畸变点云 |

### control/
| 文件 | 功能 |
|------|------|
| `mavsdk_controller.py` | MAVSDK 封装：解锁、Offboard、SendVelNED、遥测回读 |
| `action_decomposer.py` | DRL 离散动作 → NED/机体速度解算 |

### rl/
| 文件 | 功能 |
|------|------|
| `rl_agent.py` | DRL 推理接口 (支持 SB3/PyTorch 和 ONNX Runtime) |

### 测试脚本
| 文件 | 功能 |
|------|------|
| `test_live_nocontrol_raw.py` | 无飞控 raw 测试：原始 MID360 → NN-fill 深度 → ONNX DRL |
| `test_live_nocontrol.py` | 无飞控 FAST-LIO 测试：去畸变点云 → HALSS + SparseNet → DRL |

## 实时性能

验收以 Orin + Mid360 + FAST-LIO 现场日志的 P50/P95/max 为准：

| 环节 | 实现 | 验收 |
|------|------|------|
| FAST-LIO | 外部 ROS1/C++ 节点输出 `/cloud_registered` 与 `/Odometry` | `rostopic hz` 确认频率 |
| HALSS Bayesian | CPU surface-normal + CUDA UNet/MC Dropout | 日志 `H=` P50/P95 |
| 深度投影/NN-fill | 几何最近邻 + 平滑 (raw) 或 PyTorch CUDA 透视 (FAST-LIO) | 日志 `D=` P50/P95 |
| ONNX DRL | ONNX Runtime 推理，零 TF 依赖 | 日志 `RL=` P50/P95 |
| **闭环总耗时** | 串行闭环 | P95 ≤100ms，超预算帧 ≤5% |

## 故障排查

1. **FAST-LIO 无输出** → 检查话题: `rostopic list | grep -E "livox|cloud_registered|Odometry"`
2. **MID360 无数据** → 检查 IP 配置: `~/livox_ws/src/livox_ros_driver2/config/MID360_config.json` (lidar IP: 192.168.1.103, host IP: 192.168.1.5)
3. **MAVSDK 连接失败** → 检查串口权限: `sudo chmod 666 /dev/ttyACM0`
4. **ONNX 模型缺失** → 在 x86 训练机运行: `python scripts/export_ppo2_to_onnx.py`
5. **rospy 导入失败** → 确认 source: `source /opt/ros/noetic/setup.bash`
6. **OpenCV imshow 崩溃** → 设置 DISPLAY: `export DISPLAY=:0`
