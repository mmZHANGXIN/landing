#!/usr/bin/env python3
"""
测试数据源 — 模拟感知+深度补全, 直接发布 dense_depth_frame 给 DRL
=================================================================
用途: 在没有 LiDAR / 相机时测试 DRL 推理管线

产生带结构的假深度和语义图, 以 4Hz 发布到 ZeroMQ

用法:
  conda activate fylanding
  python scripts/test_zmq_source.py --pub-address tcp://127.0.0.1:5556 --rate 4
"""

import time
import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control.zmq_protocol import (
    serialize_dense_depth_frame,
    CLASS_TO_GRAY,
)

try:
    import zmq
except ImportError:
    print("pip install pyzmq")
    sys.exit(1)


def generate_frame(frame_id: int, scenario: str = "terrain_far"):
    """生成一帧模拟观测"""
    h, w = 128, 128

    if scenario == "terrain_far":
        # 远距离, 全 terrain
        depth = np.full((h, w), 25.0, dtype=np.float32)  # 25m (正值: 距离)
        sem = np.full((h, w), 1, dtype=np.uint8)

    elif scenario == "terrain_near":
        depth = np.full((h, w), 3.0, dtype=np.float32)
        sem = np.full((h, w), 1, dtype=np.uint8)

    elif scenario == "mixed":
        depth = np.random.uniform(5.0, 30.0, (h, w)).astype(np.float32)
        sem = np.full((h, w), 1, dtype=np.uint8)
        sem[:, 64:] = 9   # 右半边 danger
        depth[:, 64:] = np.random.uniform(2.0, 10.0, (h, 64)).astype(np.float32)

    elif scenario == "descend":
        depth = np.full((h, w), 1.0, dtype=np.float32)
        sem = np.full((h, w), 1, dtype=np.uint8)

    else:
        depth = np.random.uniform(5.0, 30.0, (h, w)).astype(np.float32)
        sem = np.random.choice([1, 4, 5, 9], (h, w)).astype(np.uint8)

    return depth, sem


def main():
    parser = argparse.ArgumentParser(description="Test ZMQ data source")
    parser.add_argument("--pub-address", default="tcp://127.0.0.1:5556")
    parser.add_argument("--rate", type=float, default=4.0, help="Hz")
    args = parser.parse_args()

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind(args.pub_address)
    print(f"Test source PUB → {args.pub_address} @ {args.rate} Hz")

    scenarios = ["terrain_far", "terrain_far", "mixed", "terrain_near", "descend"]
    frame_id = 0
    dt = 1.0 / args.rate

    try:
        while True:
            scenario = scenarios[frame_id % len(scenarios)]
            depth, sem = generate_frame(frame_id, scenario)

            header, payload = serialize_dense_depth_frame(frame_id, depth, sem)
            pub.send(header + payload)

            print(f"[SRC] frame={frame_id:5d}  scenario={scenario:15s}  "
                  f"depth=[{depth.min():.1f},{depth.max():.1f}]m  "
                  f"sem_unique={np.unique(sem).tolist()}  "
                  f"size={len(header)+len(payload)}B")

            frame_id += 1
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        pub.close()
        ctx.term()


if __name__ == "__main__":
    main()
