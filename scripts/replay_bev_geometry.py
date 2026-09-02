#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orin Landing — 离线对比脚本二: BEV 粗糙度保持降采样 + 几何语义分支
===================================================================
从 rosbag 读取 /cloud_registered_body 与位姿, 处理链:

  去畸变 body 点云
  → Mid360/IMU 外参变换
  → PX4 roll/pitch 水平化
  → z-up 转 z-down
  → 动态 ROI
  → BEV 粗糙度保持降采样 (与脚本一完全一致)
  → 地面高度栅格 → 局部填洞 → 局部平滑 → Sobel 梯度 → 法向量 → 坡度 → 粗糙度
  → 几何安全/危险图 (不调用 HALSSBayesianEvaluator)
  → training-camera 深度投影 (与脚本一完全一致)
  → NN-fill
  → ONNX DRL

点云与深度路径与脚本一完全一致, 仅语义分支不同.

判定规则:
  safe = slope < slope_threshold_deg and roughness < roughness_threshold
空洞区域不使用全局均值补全, 统一标记为未知/危险, 避免产生虚假的安全地面.

用法:
  python replay_bev_geometry.py \
    --bag /home/ifsc_orin/evelyn/landing/experiments/runs/20260807_162946_orin_landing/input.bag \
    --config /home/ifsc_orin/evelyn/landing/experiments/runs/20260807_162946_orin_landing/experiment_config_snapshot.yaml \
    --pose-topic /mavros/local_position/odom \
    --onnx-model weights/ppo2_policy.onnx [--no-display] [--max-frames 10]
    [--slope-threshold-deg 10.0] [--roughness-threshold-m 0.15]
"""

from __future__ import annotations

import logging

from replay_compare_common import (
    GeometrySemanticBranch,
    cfg_value,
    make_common_parser,
    perception_params,
    run_standard_replay,
    setup_logging,
)

logger = logging.getLogger("ReplayBevGeometry")


def main():
    parser = make_common_parser(
        "离线回放: BEV 粗糙度保持降采样 + 几何语义分支 (坡度/粗糙度) + "
        "training-camera 深度投影 + ONNX DRL"
    )
    parser.add_argument("--slope-threshold-deg", type=float, default=None,
                        help="坡度安全阈值 (度); 默认取配置 slope_threshold_deg "
                             "(缺省 10.0)")
    parser.add_argument("--roughness-threshold-m", type=float, default=None,
                        help="粗糙度安全阈值 (米); 默认取配置 "
                             "geometric_roughness_threshold_m (缺省 0.15)")
    args = parser.parse_args()
    setup_logging()

    def branch_factory(cfg):
        params = perception_params(cfg)
        perc = params["perc_cfg"]
        slope_th = cfg_value(args.slope_threshold_deg,
                             perc.get("slope_threshold_deg"), 10.0)
        rough_th = cfg_value(args.roughness_threshold_m,
                             perc.get("geometric_roughness_threshold_m"), 0.15)
        logger.info("[Geometry] slope_threshold=%.2f deg roughness_threshold=%.3f m",
                    slope_th, rough_th)
        return GeometrySemanticBranch(
            slope_th, rough_th,
            safe_id=params["safe_id"], danger_id=params["danger_id"],
            smooth_ksize=perc.get("geometric_smooth_ksize", 5),
        )

    run_standard_replay(args, "bev_geometry", branch_factory)


if __name__ == "__main__":
    main()
