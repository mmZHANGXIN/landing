#!/usr/bin/env python3
"""
ZeroMQ 共享协议模块
====================
定义感知→补全→DRL 三节点间的消息格式、编码/解码、版本号。

消息流向:
  Perception (PyTorch) ──PUB──> Depth Completion (TF/Keras) ──PUB──> DRL Control (TF1)

协议版本: 1.0
"""

import struct
import time
import json
import zlib
from typing import Optional, Dict, Any, Tuple
import numpy as np

# ============================================================
# 协议常量
# ============================================================

PROTOCOL_VERSION = 1
ZMQ_PROTOCOL_MAGIC = b"ORIN"

# 消息类型
MSG_TYPE_SPARSE_DEPTH = 0x01   # 感知 → 补全
MSG_TYPE_DENSE_DEPTH = 0x02    # 补全 → DRL
MSG_TYPE_HEARTBEAT = 0xFF      # 心跳/状态

# 编码标志
FLAG_NONE = 0x00
FLAG_COMPRESSED = 0x01         # 数据已用 zlib 压缩
FLAG_MULTIPART = 0x02          # 多帧拼接 (保留)

# 固定数据类型与形状
DTYPE_SPARSE_DEPTH = np.float32
DTYPE_DENSE_DEPTH = np.float32
DTYPE_VALID_MASK = np.uint8
DTYPE_SEMANTIC_ID = np.uint8

# CLASS_TO_GRAY 映射 (来自 DeepRL semantics_classes.py)
CLASS_TO_GRAY = {
    -1: 0,     # kUnknown
     0: 10,    # kPavement
     1: 30,    # kTerrain (safe)
     2: 60,    # kWater
     3: 70,    # kSky
     4: 20,    # kBuilding
     5: 40,    # kVegetation
     6: 80,    # kPerson
     7: 90,    # kRider
     8: 50,    # kVehicle
     9: 250,   # kOthers (danger)
}

# 深度编码参数 (与 DeepRL 训练时一致)
DEPTH_CLIP_MIN_M = 0.0
DEPTH_CLIP_MAX_M = 30.0
DEPTH_ENCODE_SCALE = 255.0 / DEPTH_CLIP_MAX_M  # 0~30m → 0~255

# 观测尺寸 (DeepRL PPO2 输入)
OBS_HEIGHT = 128
OBS_WIDTH = 128
OBS_CHANNELS = 2  # depth (ch0), semantic (ch1)


# ============================================================
# 消息头结构 (固定 32 字节)
# ============================================================
# magic(4) + version(1) + msg_type(1) + flags(1) + reserved(1)
# + frame_id(8) + timestamp(8) + payload_size(4) + orig_height(2) + orig_width(2)
# = 32 bytes

HEADER_FORMAT = ">4sBBBBQqIHH"  # big-endian
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


def pack_header(msg_type: int, frame_id: int, timestamp_ns: int,
                payload_size: int, orig_h: int, orig_w: int,
                flags: int = FLAG_NONE) -> bytes:
    """打包 32 字节消息头."""
    return struct.pack(HEADER_FORMAT,
                       ZMQ_PROTOCOL_MAGIC,
                       PROTOCOL_VERSION,
                       msg_type,
                       flags,
                       0,  # reserved
                       frame_id,
                       timestamp_ns,
                       payload_size,
                       orig_h,
                       orig_w)


def unpack_header(header: bytes) -> Dict[str, Any]:
    """解包 32 字节消息头, 返回字典."""
    magic, ver, msg_type, flags, _, frame_id, ts, psize, h, w = \
        struct.unpack(HEADER_FORMAT, header[:HEADER_SIZE])
    if magic != ZMQ_PROTOCOL_MAGIC:
        raise ValueError(f"Bad magic: {magic!r}, expected {ZMQ_PROTOCOL_MAGIC!r}")
    if ver != PROTOCOL_VERSION:
        raise ValueError(f"Protocol version mismatch: got {ver}, expected {PROTOCOL_VERSION}")
    return {
        "magic": magic,
        "version": ver,
        "msg_type": msg_type,
        "flags": flags,
        "frame_id": frame_id,
        "timestamp_ns": ts,
        "payload_size": psize,
        "orig_height": h,
        "orig_width": w,
    }


# ============================================================
# 序列化 / 反序列化
# ============================================================

def serialize_sparse_depth_frame(
    frame_id: int,
    sparse_depth: np.ndarray,   # HxW float32, meters
    valid_mask: np.ndarray,     # HxW uint8, 1=valid
    semantic_id: np.ndarray,    # HxW uint8, class IDs
    pose: np.ndarray = None,    # optional [x, y, z, qw, qx, qy, qz]
    camera_intrinsics: dict = None,
    depth_scale: float = 1.0,
    compress: bool = False,
) -> Tuple[bytes, bytes]:
    """
    序列化稀疏深度帧 → (header, payload)

    Payload 格式 (小端 packed):
      depth_scale (4B float32)
      meta_json_len (4B uint32) + meta_json (UTF-8)
      sparse_depth (H*W*4B float32)
      valid_mask (H*W*1B uint8)
      semantic_id (H*W*1B uint8)

    返回 (header_bytes, payload_bytes)
    """
    h, w = sparse_depth.shape[:2]
    sd = np.ascontiguousarray(sparse_depth, dtype=np.float32)
    vm = np.ascontiguousarray(valid_mask, dtype=np.uint8)
    si = np.ascontiguousarray(semantic_id, dtype=np.uint8)

    # 元数据 JSON
    meta = {
        "depth_scale": float(depth_scale),
    }
    if pose is not None:
        meta["pose"] = pose.tolist() if hasattr(pose, "tolist") else list(pose)
    if camera_intrinsics is not None:
        meta["camera_intrinsics"] = camera_intrinsics

    meta_bytes = json.dumps(meta).encode("utf-8")

    # 拼接 payload
    payload = b""
    payload += struct.pack("<f", depth_scale)
    payload += struct.pack("<I", len(meta_bytes))
    payload += meta_bytes
    payload += sd.tobytes()
    payload += vm.tobytes()
    payload += si.tobytes()

    flags = FLAG_COMPRESSED if compress else FLAG_NONE
    if compress:
        payload = zlib.compress(payload)

    timestamp_ns = int(time.time() * 1e9)
    header = pack_header(MSG_TYPE_SPARSE_DEPTH, frame_id, timestamp_ns,
                         len(payload), h, w, flags)
    return header, payload


def deserialize_sparse_depth_frame(header_bytes: bytes, payload_bytes: bytes) -> Dict[str, Any]:
    """
    反序列化 → 返回字典:
      frame_id, timestamp_ns, sparse_depth, valid_mask, semantic_id,
      depth_scale, pose, camera_intrinsics
    """
    hdr = unpack_header(header_bytes)
    payload = payload_bytes
    if hdr["flags"] & FLAG_COMPRESSED:
        payload = zlib.decompress(payload)

    offset = 0
    depth_scale = struct.unpack_from("<f", payload, offset)[0]; offset += 4
    meta_len = struct.unpack_from("<I", payload, offset)[0]; offset += 4
    meta = json.loads(payload[offset:offset + meta_len].decode("utf-8")); offset += meta_len

    h, w = hdr["orig_height"], hdr["orig_width"]
    sd_size = h * w * 4
    vm_size = h * w * 1
    si_size = h * w * 1

    sparse_depth = np.frombuffer(payload, dtype=np.float32, count=h*w, offset=offset).reshape(h, w).copy()
    offset += sd_size
    valid_mask = np.frombuffer(payload, dtype=np.uint8, count=h*w, offset=offset).reshape(h, w).copy()
    offset += vm_size
    semantic_id = np.frombuffer(payload, dtype=np.uint8, count=h*w, offset=offset).reshape(h, w).copy()

    result = {
        "frame_id": hdr["frame_id"],
        "timestamp_ns": hdr["timestamp_ns"],
        "sparse_depth": sparse_depth,
        "valid_mask": valid_mask,
        "semantic_id": semantic_id,
        "depth_scale": depth_scale,
    }
    if "pose" in meta:
        result["pose"] = np.array(meta["pose"], dtype=np.float64)
    if "camera_intrinsics" in meta:
        result["camera_intrinsics"] = meta["camera_intrinsics"]
    return result


def serialize_dense_depth_frame(
    frame_id: int,
    dense_depth: np.ndarray,   # HxW float32, meters
    semantic_id: np.ndarray,   # HxW uint8, class IDs
    compress: bool = False,
) -> Tuple[bytes, bytes]:
    """
    序列化稠密深度帧 → (header, payload)

    Payload 格式:
      dense_depth (H*W*4B float32)
      semantic_id (H*W*1B uint8)

    返回 (header_bytes, payload_bytes)
    """
    h, w = dense_depth.shape[:2]
    dd = np.ascontiguousarray(dense_depth, dtype=np.float32)
    si = np.ascontiguousarray(semantic_id, dtype=np.uint8)

    payload = dd.tobytes() + si.tobytes()

    flags = FLAG_COMPRESSED if compress else FLAG_NONE
    if compress:
        payload = zlib.compress(payload)

    timestamp_ns = int(time.time() * 1e9)
    header = pack_header(MSG_TYPE_DENSE_DEPTH, frame_id, timestamp_ns,
                         len(payload), h, w, flags)
    return header, payload


def deserialize_dense_depth_frame(header_bytes: bytes, payload_bytes: bytes) -> Dict[str, Any]:
    """反序列化稠密深度帧."""
    hdr = unpack_header(header_bytes)
    payload = payload_bytes
    if hdr["flags"] & FLAG_COMPRESSED:
        payload = zlib.decompress(payload)

    h, w = hdr["orig_height"], hdr["orig_width"]
    dd_size = h * w * 4

    dense_depth = np.frombuffer(payload, dtype=np.float32, count=h*w, offset=0).reshape(h, w).copy()
    semantic_id = np.frombuffer(payload, dtype=np.uint8, count=h*w, offset=dd_size).reshape(h, w).copy()

    return {
        "frame_id": hdr["frame_id"],
        "timestamp_ns": hdr["timestamp_ns"],
        "dense_depth": dense_depth,
        "semantic_id": semantic_id,
    }


def serialize_heartbeat(node_name: str, status: str, fps: float = 0.0) -> bytes:
    """序列化心跳消息."""
    meta = json.dumps({"node": node_name, "status": status, "fps": fps}).encode("utf-8")
    timestamp_ns = int(time.time() * 1e9)
    header = pack_header(MSG_TYPE_HEARTBEAT, 0, timestamp_ns, len(meta), 0, 0)
    return header + meta


# ============================================================
# DRL 观测适配 (深度 + 语义 → 128x128x2 uint8)
# ============================================================

def adapt_to_drl_observation(
    dense_depth_m: np.ndarray,
    semantic_id: np.ndarray,
    dmax: float = 30.0,
) -> np.ndarray:
    """
    将稠密深度 (米) + 语义类别 ID 转换为 DeepRL PPO2 兼容的 (128,128,2) uint8 观测。

    编码规则 (与训练时 Strict 兼容):
      - channel 0 = depth:  clip(0, dmax) / dmax * 255  → uint8
      - channel 1 = semantic: CLASS_TO_GRAY 映射 → uint8

    参数:
      dense_depth_m: HxW float32, 深度 (米)
      semantic_id:   HxW uint8, 语义类别 ID (0-9)
      dmax: 深度上限 (米), 默认 30

    返回:
      obs: (128, 128, 2) np.uint8
    """
    import cv2

    # Resize 到 128x128
    if dense_depth_m.shape[:2] != (OBS_HEIGHT, OBS_WIDTH):
        dd = cv2.resize(dense_depth_m, (OBS_WIDTH, OBS_HEIGHT),
                        interpolation=cv2.INTER_LINEAR)
    else:
        dd = dense_depth_m

    if semantic_id.shape[:2] != (OBS_HEIGHT, OBS_WIDTH):
        si = cv2.resize(semantic_id, (OBS_WIDTH, OBS_HEIGHT),
                        interpolation=cv2.INTER_NEAREST)
    else:
        si = semantic_id

    # Depth channel: clip + scale → 0-255 uint8
    dd_clipped = np.clip(dd, 0.0, dmax)
    depth_ch = (dd_clipped / dmax * 255.0).astype(np.uint8)

    # Semantic channel: CLASS_TO_GRAY lookup
    sem_gray = np.zeros_like(si, dtype=np.uint8)
    for class_id, gray_val in CLASS_TO_GRAY.items():
        sem_gray[si == class_id] = gray_val

    # Stack: channel 0 = depth, channel 1 = semantic
    obs = np.stack([depth_ch, sem_gray], axis=-1)  # (128, 128, 2)
    return obs


def validate_observation(obs: np.ndarray) -> bool:
    """验证观测是否符合 DeepRL 预期."""
    if obs.shape != (OBS_HEIGHT, OBS_WIDTH, OBS_CHANNELS):
        return False
    if obs.dtype != np.uint8:
        return False
    return True
