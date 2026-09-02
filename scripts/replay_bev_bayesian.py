#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orin Landing — 离线对比脚本一: BEV 粗糙度保持降采样 + HALSS Bayesian
=====================================================================
从 rosbag 读取 /cloud_registered_body (FAST-LIO 去畸变 body 点云) 与位姿
(/mavros/local_position/odom), 处理链:

  去畸变 body 点云
  → Mid360/IMU 外参变换
  → PX4 roll/pitch 水平化
  → z-up 转 z-down
  → 动态 ROI
  → BEV 粗糙度保持降采样 (128 网格, 每单元保留 z-down 最大点/最小点/高度差/点数)
  → HALSS Bayesian (UNet + MC Dropout)
  → 语义图
  → training-camera 深度投影 (与 HALSS 共用同一批 BEV 点)
  → NN-fill
  → ONNX DRL

取消「每条针孔射线只保留最近点」采样: 对比仅反映 BEV 降采样方式变化.
/livox/lidar (CustomMsg) 仅作原始点云参考统计与可选可视化, 不参与推理.

用法:
  python replay_bev_bayesian.py \
    --bag /home/ifsc_orin/evelyn/landing/experiments/runs/20260807_162946_orin_landing/input.bag \
    --config /home/ifsc_orin/evelyn/landing/experiments/runs/20260807_162946_orin_landing/experiment_config_snapshot.yaml \
    --pose-topic /mavros/local_position/odom \
    --onnx-model weights/ppo2_policy.onnx [--no-display] [--max-frames 10]

依赖: HALSS Bayesian 需要 torch + CUDA (配置 require_gpu); 无 torch/CUDA
环境运行本脚本会明确报错, 请改用 replay_bev_geometry.py.
"""

from __future__ import annotations

import logging

from replay_compare_common import (
    BayesianSemanticBranch,
    make_common_parser,
    perception_params,
    run_standard_replay,
    setup_logging,
)

logger = logging.getLogger("ReplayBevBayesian")


def main():
    parser = make_common_parser(
        "离线回放: BEV 粗糙度保持降采样 + HALSS Bayesian 语义 + "
        "training-camera 深度投影 + ONNX DRL"
    )
    args = parser.parse_args()
    setup_logging()

    def branch_factory(cfg):
        params = perception_params(cfg)
        return BayesianSemanticBranch(
            cfg,
            obs_w=params["obs_w"], obs_h=params["obs_h"],
            danger_id=params["danger_id"],
        )

    run_standard_replay(args, "bev_bayesian", branch_factory)


if __name__ == "__main__":
    main()
