#!/usr/bin/env python3
"""
SB2 (stable-baselines) → SB3 (stable-baselines3) PPO 权重转换器

SB2 序列化格式:
  - parameter_list: JSON 数组, TF 变量名列表
  - parameters:     嵌套 ZIP 包, 包含 .npy 文件 (按 parameter_list 索引)
  - data:           JSON, 训练元数据

用法:
    source venv/bin/activate
    python convert_sb2_to_sb3.py \
        --input ../FaultyYawLanding/DeepRL/data/trained_policy/last_step_model.zip \
        --output weights/last_step_model_sb3_legacy_test.zip \
        --allow-legacy-conversion

注意:
  当前飞控使用的 weights/last_step_model_sb3.zip 是 HWC(128,128,2),
  normalize_images=false, custom SB2CNN 的模型包。此脚本是早期转换器,
  默认会被保护性拒绝执行, 防止覆盖已验收权重。
"""

import argparse
import json
import os
import sys
import tempfile
import zipfile


def extract_sb2_weights(zip_path: str) -> dict:
    """
    从 SB2 的 zip 文件中提取所有权重。
    SB2 格式: parameter_list (JSON) + parameters (嵌套zip含.npy)
    返回: {variable_name: numpy_array}
    """
    tmpdir = tempfile.mkdtemp(prefix="sb2_extract_")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmpdir)

        # 1. 读取 parameter_list (JSON)
        with open(os.path.join(tmpdir, "parameter_list"), "r") as f:
            param_names = json.load(f)
        print(f"[Extract] Found {len(param_names)} parameters in SB2 model")

        # 2. 读取 parameters (嵌套 ZIP, 内含 .npy)
        weights = {}
        with zipfile.ZipFile(os.path.join(tmpdir, "parameters"), "r") as inner_zip:
            import numpy as np
            npy_files = sorted(inner_zip.namelist())
            print(f"[Extract] Found {len(npy_files)} .npy files in parameters archive")

            for i, npy_name in enumerate(npy_files):
                tf_name = param_names[i] if i < len(param_names) else npy_name
                with inner_zip.open(npy_name) as npy_f:
                    arr = np.load(npy_f)
                weights[tf_name] = arr

        # 打印所有参数
        for name in sorted(weights.keys()):
            print(f"  {name}: shape={weights[name].shape}, dtype={weights[name].dtype}")

        return weights

    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def map_sb2_ppo_weights_to_sb3(sb2_weights: dict, obs_shape: tuple,
                                 act_dim: int) -> dict:
    """
    将 SB2 PPO2 的 TF 权重映射到 SB3 PPO 的 PyTorch state_dict。

    实际 SB2 变量命名 (从 parameter_list 提取):
      model/c1/w:0, model/c1/b:0  → CNN conv1
      model/c2/w:0, model/c2/b:0  → CNN conv2
      model/c3/w:0, model/c3/b:0  → CNN conv3
      model/fc1/w:0, model/fc1/b:0 → 共享 FC (展平后)
      model/pi_fc0/kernel:0, bias:0 → policy hidden layer 0
      model/pi_fc1/kernel:0, bias:0 → policy hidden layer 1
      model/pi_fc2/kernel:0, bias:0 → policy hidden layer 2
      model/vf_fc0/kernel:0, bias:0 → value hidden layer 0
      model/vf_fc1/kernel:0, bias:0 → value hidden layer 1
      model/pi/w:0, model/pi/b:0    → action output
      model/vf/kernel:0, bias:0     → value output
      model/q/w:0, model/q/b:0      → optional q-value output
    """
    import numpy as np
    import torch

    state_dict = {}

    # ---- CNN Feature Extractor (3 conv layers) ----
    # SB2 CNN kernel: (H, W, C_in, C_out) → PyTorch: (C_out, C_in, H, W)
    for i, layer in enumerate(["c1", "c2", "c3"]):
        w_key = f"model/{layer}/w:0"
        b_key = f"model/{layer}/b:0"
        if w_key in sb2_weights:
            w = sb2_weights[w_key]
            b = sb2_weights.get(b_key, np.zeros(w.shape[-1]))
            w_pt = np.transpose(w, (3, 2, 0, 1)).copy()
            state_dict[f"features_extractor.cnn.{i}.weight"] = torch.from_numpy(w_pt)
            state_dict[f"features_extractor.cnn.{i}.bias"] = torch.from_numpy(b.copy())

    # ---- 共享 FC (展平后的第一个全连接) ----
    shared_w_key = "model/fc1/w:0"
    shared_b_key = "model/fc1/b:0"
    if shared_w_key in sb2_weights:
        w = sb2_weights[shared_w_key]  # TF: (in, out)
        b = sb2_weights.get(shared_b_key, np.zeros(w.shape[-1]))
        w_pt = np.transpose(w, (1, 0)).copy()  # → (out, in)
        # SB3: mlp_extractor.shared_net.0
        state_dict["mlp_extractor.shared_net.0.weight"] = torch.from_numpy(w_pt)
        state_dict["mlp_extractor.shared_net.0.bias"] = torch.from_numpy(b.copy())
        print(f"  [Map] Shared FC: {w_pt.shape}")

    # ---- Policy MLP (pi_fc0, pi_fc1, pi_fc2) ----
    for i in range(3):
        w_key = f"model/pi_fc{i}/kernel:0"
        b_key = f"model/pi_fc{i}/bias:0"
        if w_key in sb2_weights:
            w = sb2_weights[w_key]
            b = sb2_weights.get(b_key, np.zeros(w.shape[-1]))
            w_pt = np.transpose(w, (1, 0)).copy()
            state_dict[f"mlp_extractor.policy_net.{i}.weight"] = torch.from_numpy(w_pt)
            state_dict[f"mlp_extractor.policy_net.{i}.bias"] = torch.from_numpy(b.copy())

    # ---- Value MLP (vf_fc0, vf_fc1) ----
    for i in range(2):
        w_key = f"model/vf_fc{i}/kernel:0"
        b_key = f"model/vf_fc{i}/bias:0"
        if w_key in sb2_weights:
            w = sb2_weights[w_key]
            b = sb2_weights.get(b_key, np.zeros(w.shape[-1]))
            w_pt = np.transpose(w, (1, 0)).copy()
            state_dict[f"mlp_extractor.value_net.{i}.weight"] = torch.from_numpy(w_pt)
            state_dict[f"mlp_extractor.value_net.{i}.bias"] = torch.from_numpy(b.copy())

    # ---- Action output: model/pi/w:0 ----
    if "model/pi/w:0" in sb2_weights:
        w = sb2_weights["model/pi/w:0"]
        b = sb2_weights.get("model/pi/b:0", np.zeros(w.shape[-1]))
        w_pt = np.transpose(w, (1, 0)).copy()
        state_dict["action_net.weight"] = torch.from_numpy(w_pt)
        state_dict["action_net.bias"] = torch.from_numpy(b.copy())

    # ---- Value output: model/vf/kernel:0 ----
    if "model/vf/kernel:0" in sb2_weights:
        w = sb2_weights["model/vf/kernel:0"]
        b = sb2_weights.get("model/vf/bias:0", np.zeros(w.shape[-1]))
        w_pt = np.transpose(w, (1, 0)).copy()
        state_dict["value_net.weight"] = torch.from_numpy(w_pt)
        state_dict["value_net.bias"] = torch.from_numpy(b.copy())

    # ---- Log std (SB2 若存在) ----
    logstd_key = "model/pi/logstd:0"
    if logstd_key in sb2_weights:
        logstd_val = float(sb2_weights[logstd_key])
        state_dict["action_net_log_std"] = torch.full((act_dim,), logstd_val)
    else:
        # SB3 需要 log_std
        state_dict["action_net_log_std"] = torch.full((act_dim,), -0.5)

    print(f"[Map] Mapped {len(state_dict)} parameter tensors to SB3 format")
    return state_dict


def main():
    parser = argparse.ArgumentParser(description="SB2 PPO → SB3 PPO 权重转换")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to SB2 last_step_model.zip")
    parser.add_argument("--output", type=str, default="weights/last_step_model_sb3.zip",
                        help="Output path for SB3 model")
    parser.add_argument("--img-channels", type=int, default=2,
                        help="Number of input image channels (depth + semantics = 2)")
    parser.add_argument("--img-width", type=int, default=128)
    parser.add_argument("--img-height", type=int, default=128)
    parser.add_argument("--act-dim", type=int, default=10,
                        help="Action space dimension (discrete)")
    parser.add_argument("--allow-legacy-conversion", action="store_true",
                        help="Run this legacy converter despite known metadata mismatch risks")
    args = parser.parse_args()

    if not args.allow_legacy_conversion:
        print("[ABORT] This is the legacy SB2→SB3 converter.")
        print("        It does not reproduce the current flight policy metadata exactly.")
        print("        Use inspect_drl_model.py to verify the active policy, and pass")
        print("        --allow-legacy-conversion only when writing to a test output path.")
        sys.exit(2)
    active_policy = os.path.abspath("weights/last_step_model_sb3.zip")
    if os.path.abspath(args.output) == active_policy:
        print("[ABORT] Refusing to overwrite active flight policy weights/last_step_model_sb3.zip.")
        print("        Write to weights/last_step_model_sb3_legacy_test.zip and validate first.")
        sys.exit(2)

    import numpy as np
    from stable_baselines3 import PPO
    from gymnasium import spaces

    print("=" * 60)
    print(" SB2 PPO → SB3 PPO 权重转换器")
    print("=" * 60)

    # 1. 提取 SB2 权重
    print(f"\n[Step 1] Extracting SB2 weights from: {args.input}")
    sb2_weights = extract_sb2_weights(args.input)

    # 2. 映射到 SB3 格式
    print(f"\n[Step 2] Mapping weights to SB3 format...")
    obs_shape = (args.img_channels, args.img_height, args.img_width)
    state_dict = map_sb2_ppo_weights_to_sb3(sb2_weights, obs_shape, args.act_dim)

    if len(state_dict) == 0:
        print("\n[ERROR] No weights mapped! Check if the SB2 model uses expected naming.")
        print("Available keys:", sorted(sb2_weights.keys()))
        sys.exit(1)

    # 3. 创建 SB3 PPO 模型并加载权重
    print(f"\n[Step 3] Creating SB3 PPO model...")
    observation_space = spaces.Box(
        low=0.0, high=1.0,
        shape=obs_shape, dtype=np.float32
    )
    action_space = spaces.Discrete(args.act_dim)

    # SB2 实际架构: CNN → shared_fc → pi_fc0→pi_fc1→pi_fc2 + vf_fc0→vf_fc1
    # SB3 net_arch: dict(pi=[...], vf=[...]) 如果有 shared_net 则用 share_features_extractor=True
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256, 256], vf=[256, 256]),
        share_features_extractor=True,  # 对应 SB2 的 model/fc1 共享层
    )

    model = PPO(
        "CnnPolicy",
        observation_space,
        action_space,
        policy_kwargs=policy_kwargs,
        verbose=0,
    )

    # 4. 加载映射后的权重
    print(f"\n[Step 4] Loading mapped weights into SB3 model...")
    model_state = model.policy.state_dict()
    loaded = 0
    skipped = []

    for key, tensor in state_dict.items():
        if key in model_state:
            if model_state[key].shape == tensor.shape:
                model_state[key] = tensor
                loaded += 1
            else:
                skipped.append(f"{key}: SB3 shape {model_state[key].shape} ≠ SB2 shape {tensor.shape}")
        else:
            skipped.append(f"{key}: not found in SB3 model")

    model.policy.load_state_dict(model_state, strict=False)

    print(f"  Loaded: {loaded} parameters")
    if skipped:
        print(f"  Skipped: {len(skipped)} parameters")
        for s in skipped[:5]:
            print(f"    - {s}")
        if len(skipped) > 5:
            print(f"    ... and {len(skipped)-5} more")

    # 5. 保存 SB3 模型
    print(f"\n[Step 5] Saving SB3 model to: {args.output}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    model.save(args.output)
    print(f"  Model saved! File size: {os.path.getsize(args.output) / 1024:.1f} KB")

    # 6. 验证
    print(f"\n[Step 6] Verifying converted model...")
    loaded_model = PPO.load(args.output)
    dummy_obs = np.random.randn(1, *obs_shape).astype(np.float32)
    action, _ = loaded_model.predict(dummy_obs)
    print(f"  Test inference with random input: action = {action}")
    print(f"  ✅ Conversion successful!")


if __name__ == "__main__":
    main()
