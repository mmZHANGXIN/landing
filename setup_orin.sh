#!/bin/bash
# ============================================================
# Orin Landing - Jetson Orin NX 16GB 环境搭建脚本 (ROS1 Noetic)
# 需要: Ubuntu 20.04, ROS Noetic, CUDA 11.4 (JetPack)
# ============================================================
set -e

echo "=============================================="
echo " Orin Landing - Environment Setup"
echo " Jetson Orin NX 16GB / CUDA 11.4 / ROS1 Noetic"
echo "=============================================="

# ---- Step 1: 系统基础包 ----
echo "[1/7] Installing system dependencies..."
sudo apt-get update -y
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    libopenblas-dev \
    libopenblas-base \
    libsm6 \
    libxrender-dev \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-tk \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev

# ---- Step 2: Python 虚拟环境 ----
echo "[2/7] Creating Python virtual environment..."
python3 -m venv venv --system-site-packages
source venv/bin/activate

# ---- Step 3: PyTorch (Jetson 专用) ----
echo "[3/7] Installing PyTorch for Jetson (CUDA 11.4)..."
pip install --upgrade pip setuptools wheel
pip install numpy==1.22.4  # 先装兼容版本的 numpy
pip install torch torchvision --index-url https://download.pytorch.org/whl/jetson

# ---- Step 4: Python 核心依赖 ----
echo "[4/7] Installing Python packages..."
pip install scipy opencv-python pyyaml pandas pillow matplotlib
pip install stable-baselines3 gym

# ---- Step 5: MAVSDK ----
echo "[5/7] Installing MAVSDK..."
pip install mavsdk aioconsole

# ---- Step 6: ROS1 依赖 (从系统包) ----
echo "[6/7] Checking ROS1 Noetic environment..."
if [ -f /opt/ros/noetic/setup.bash ]; then
    source /opt/ros/noetic/setup.bash
    echo "  ROS1 Noetic found."
else
    echo "  WARNING: ROS1 Noetic not found. Install: sudo apt install ros-noetic-desktop"
fi

# ---- Step 7: 验证 ----
echo "[7/7] Verifying installation..."
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python3 -c "import cv2; print(f'OpenCV {cv2.__version__}')"
python3 -c "import numpy; print(f'NumPy {numpy.__version__}')"

echo ""
echo "=============================================="
echo " Setup complete!"
echo " Activate: source venv/bin/activate"
echo "=============================================="

# ============================================================
# ZMQ 解耦管线 — 三环境 setup (手动执行)
# ============================================================
# 以下三个 conda 环境分别对应感知、补全、决策:
#
# 1. orin_perception_pytorch (Python 3.8, PyTorch):
#    conda create -n orin_perception_pytorch python=3.8 -y
#    conda activate orin_perception_pytorch
#    pip install -r requirements_perception_pytorch.txt
#
# 2. orin_depth_tfkeras (Python 3.8, TF2/Keras):
#    conda create -n orin_depth_tfkeras python=3.8 -y
#    conda activate orin_depth_tfkeras
#    pip install -r requirements_depth_tfkeras.txt
#
# 3. drl_onnx (Python 3.8, ONNX Runtime — 无需 TF!):
#    conda create -n drl_onnx python=3.8 -y
#    conda activate drl_onnx
#    pip install numpy opencv-python pyzmq onnxruntime pyyaml
#
# ONNX 模型需先在 x86 机器导出 (一次性):
#    python scripts/export_ppo2_to_onnx.py \
#        --model arch/DeepRL/data/trained_policy/last_step_model.zip \
#        --output weights/ppo2_policy.onnx
#
# 系统依赖 (所有环境共用):
#   sudo apt install build-essential cmake libeigen3-dev
#   sudo apt install libyaml-cpp-dev libopencv-dev libpcl-dev
#   sudo apt install liboctomap-dev libgoogle-glog-dev libglm-dev libvulkan-dev
#   sudo apt install libopenmpi-dev zlib1g-dev libzmq3-dev
# ============================================================"
