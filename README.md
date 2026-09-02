# Landing

> Jetson Orin NX 16GB + Mid360 LiDAR


## Dependencies

Jetson Orin NX 16GB | ARM64, CUDA Cores + Tensor Cores |
Ubuntu 20.04 + JetPack 5.x | CUDA 11.4, cuDNN 8.5, TensorRT 8.5 |
| ROS | **Noetic** (ROS1) | FAST-LIO + livox_ros_driver2 |
| Python 3.8 |
| PyTorch | 2.1+ (Jetson版) | NVIDIA Jetson wheel |
| OpenCV | 4.5+ |
| ONNX Runtime | 1.17+ |


## Deployment

### Step 1

```bash
uname -m
nvcc --version
source /opt/ros/noetic/setup.bash && echo "ROS Noetic OK"
```

### Step 2

```bash
cd ~
git clone https://github.com/Livox-SDK/Livox-SDK2.git
cd Livox-SDK2 && mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(nproc) && sudo make install

mkdir -p ~/livox_ws/src
cd ~/livox_ws/src
git clone https://github.com/Livox-SDK/livox_ros_driver2.git
cd livox_ros_driver2
source /opt/ros/noetic/setup.bash
./build.sh ROS1

# MID360 IP
vim ~/livox_ws/src/livox_ros_driver2/config/MID360_config.json
```

### Step 3

```bash
mkdir -p ~/fast_lio_ws/src
cp -r ~/evelyn/landing/arch/FAST_LIO ~/fast_lio_ws/src/
cd ~/fast_lio_ws/src/FAST_LIO
sed -i 's/livox_ros_driver</livox_ros_driver2</g' package.xml
sed -i 's/livox_ros_driver/livox_ros_driver2/g' CMakeLists.txt
find . -name "*.h" -o -name "*.cpp" | xargs sed -i \
  's|livox_ros_driver/|livox_ros_driver2/|g; s|livox_ros_driver::|livox_ros_driver2::|g'
git clone --depth 1 --branch fast_lio \
  https://github.com/hku-mars/ikd-Tree.git include/ikd-Tree

cd ~/fast_lio_ws
source /opt/ros/noetic/setup.bash
source ~/livox_ws/devel/setup.bash
catkin_make
```

### Step 4

`config/experiment_config.yaml`:
- `uav.yaw_rate_rad_s` 
- `uav.action_lateral_sign`

### Step 5

```bash
roscore &

# MID360 CustomMsg
sudo ip addr add 192.168.1.5/24 dev eth0

source /opt/ros/noetic/setup.bash && source ~/livox_ws/devel/setup.bash
roslaunch livox_ros_driver2 msg_MID360.launch rviz_enable:=true

# FAST-LIO
source ~/fast_lio_ws/devel/setup.bash
roslaunch fast_lio mapping_mid360.launch rviz:=false
roslaunch fast_lio frontend_mid360.launch

## indoor：FAST-LIO → /mavros/vision_pose/pose → PX4 EKF2
roslaunch fast_lio mapping_mid360.launch external_vision:=true rviz:=false
python pipeline.py --config ./config/experiment_indoor_fastlio.yaml --mode ros

## outdoor：GPS → PX4 EKF2
roslaunch fast_lio mapping_mid360.launch external_vision:=false rviz:=false
python pipeline.py --config ./config/experiment_outdoor_gps.yaml --mode ros

roslaunch mavros px4.launch 
