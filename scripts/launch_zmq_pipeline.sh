#!/bin/bash
# ============================================================
# launch_zmq_pipeline.sh
# 一键启动感知→补全→DRL 三节点 ZeroMQ 管线
#
# 使用方法:
#   chmod +x launch_zmq_pipeline.sh
#   ./launch_zmq_pipeline.sh [--offline] [--validate]
#
# 环境要求 (需预先创建 conda environments):
#   orin_perception_pytorch  — PyTorch + OpenCV + HALSS
#   orin_depth_tfkeras       — TF2/Keras + Sparsity-Invariant-CNNs
#   orin_drl_onnx            — ONNX Runtime (Python 3.8, 推荐 Orin 推理)
#
# 参数:
#   --offline   离线单帧测试模式 (不进入循环)
#   --validate  运行验证脚本后退出
#   --noop      深度补全直通模式 (跳过 SparseNet)
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ---- 默认地址 ----
PERCEPTION_PUB="tcp://127.0.0.1:5555"
DEPTH_COMPLETION_SUB="tcp://127.0.0.1:5555"
DEPTH_COMPLETION_PUB="tcp://127.0.0.1:5556"
DRL_CONTROL_SUB="tcp://127.0.0.1:5556"

# ---- 默认权重路径 ----
HALSS_WEIGHT="${PROJECT_DIR}/arch/3.UDPDirect30Hz_cyd_final/HALO-master (2)/HALSS/network_utils/unet_epoch6.pth"
SPARSENET_WEIGHT="${PROJECT_DIR}/arch/Sparsity-Invariant-CNNs-master/checkpoints/sparsenet.ckpt"
PPO2_MODEL="${PROJECT_DIR}/arch/DeepRL/data/trained_policy/last_step_model.zip"

# ---- 解析参数 ----
MODE="live"
NOOP="false"
VALIDATE="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --offline)
            MODE="offline"
            shift
            ;;
        --validate)
            VALIDATE="true"
            shift
            ;;
        --noop)
            NOOP="true"
            shift
            ;;
        *)
            echo "Unknown arg: $1"
            shift
            ;;
    esac
done

echo "=============================================="
echo " Orin ZMQ Pipeline Launcher"
echo "=============================================="
echo " Project dir: ${PROJECT_DIR}"
echo " Mode:        ${MODE}"
echo " Noop pass:   ${NOOP}"
echo " Validate:    ${VALIDATE}"
echo "=============================================="

# ---- 清理函数 ----
cleanup() {
    echo ""
    echo "Shutting down pipeline..."
    # 杀死所有后台 python 进程
    if [ -n "$PID_PERCEPTION" ]; then kill "$PID_PERCEPTION" 2>/dev/null || true; fi
    if [ -n "$PID_COMPLETION" ]; then kill "$PID_COMPLETION" 2>/dev/null || true; fi
    if [ -n "$PID_DRL" ]; then kill "$PID_DRL" 2>/dev/null || true; fi
    wait 2>/dev/null
    echo "All nodes stopped."
}
trap cleanup EXIT INT TERM

# ================================================================
# 验证模式
# ================================================================
if [ "$VALIDATE" == "true" ]; then
    echo "[VALIDATE] Running offline validation..."
    python3 "${PROJECT_DIR}/scripts/validate_zmq_pipeline.py" \
        --halss-weight "${HALSS_WEIGHT}" \
        --sparsenet-weight "${SPARSENET_WEIGHT}" \
        --ppo2-model "${PPO2_MODEL}"
    exit $?
fi

# ================================================================
# 启动节点
# ================================================================

# --- Node 3: DRL Control (ONNX) --- 先启动, 等待连接
echo ""
echo "[1/3] Starting DRL Control Service (ONNX)..."

ONNX_MODEL="${PROJECT_DIR}/weights/ppo2_policy.onnx"
ONNX_META="${PROJECT_DIR}/weights/ppo2_policy_meta.json"

if [ -f "${ONNX_MODEL}" ]; then
    conda run -n drl_onnx python3 "${PROJECT_DIR}/control/drl_control_service_onnx.py" \
        --onnx-model "${ONNX_MODEL}" \
        --onnx-meta "${ONNX_META}" \
        --sub-address "${DRL_CONTROL_SUB}" \
        --vel-lateral 1.0 \
        --vel-vertical 1.0 \
        --dmax 30.0 \
        --watchdog-ms 500 \
        --fc-mode noop &
    PID_DRL=$!
else
    echo "  WARNING: ONNX model not found at ${ONNX_MODEL}"
    echo "  Export on x86 first: python scripts/export_ppo2_to_onnx.py"
    exit 1
fi
sleep 2

# --- Node 2: Depth Completion (TF/Keras) ---
echo "[2/3] Starting Depth Completion Service (TF/Keras)..."

NOOP_FLAG=""
if [ "$NOOP" == "true" ]; then
    NOOP_FLAG="--high-pass-noop"
fi

conda run -n orin_depth_tfkeras python3 "${PROJECT_DIR}/control/depth_completion_service_tf.py" \
    --sub-address "${DEPTH_COMPLETION_SUB}" \
    --pub-address "${DEPTH_COMPLETION_PUB}" \
    --sparsenet-weight "${SPARSENET_WEIGHT}" \
    --input-size 128 \
    --dmax 30.0 \
    ${NOOP_FLAG} &
PID_COMPLETION=$!
sleep 2

# --- Node 1: Perception Publisher (PyTorch) ---
echo "[3/3] Starting Perception Publisher (PyTorch)..."

if [ "$MODE" == "offline" ]; then
    # 离线单帧测试
    echo "[OFFLINE] Running single-frame test..."
    conda run -n orin_perception_pytorch python3 "${PROJECT_DIR}/control/perception_publisher.py" \
        --pub-address "${PERCEPTION_PUB}" \
        --halss-weight "${HALSS_WEIGHT}" \
        --mode offline
else
    conda run -n orin_perception_pytorch python3 "${PROJECT_DIR}/control/perception_publisher.py" \
        --pub-address "${PERCEPTION_PUB}" \
        --halss-weight "${HALSS_WEIGHT}" \
        --mode live &
    PID_PERCEPTION=$!
fi

echo ""
echo "=============================================="
echo " Pipeline running. Press Ctrl+C to stop."
echo "=============================================="

# 等待任意节点退出
wait
