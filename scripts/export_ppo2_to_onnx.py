#!/usr/bin/env python3
"""
PPO2 策略网络 → ONNX 导出脚本
==============================
在 x86 机器上运行 (Python 3.6 + TF1.15 + stable-baselines v2)

用法:
  conda activate drl_export_tf1
  python scripts/export_ppo2_to_onnx.py \
      --model arch/DeepRL/data/trained_policy/last_step_model.zip \
      --output weights/ppo2_policy.onnx

输出:
  weights/ppo2_policy.onnx       — ONNX 模型文件
  weights/ppo2_policy_meta.json  — 输入/输出节点名、obs shape 等元数据

x86 导出环境:
  conda create -n drl_export_tf1 python=3.6 -y
  conda activate drl_export_tf1
  pip install "pip<21" "setuptools<50" "wheel<0.37"
  pip install tensorflow==1.15.0
  pip install stable-baselines==2.10.2
  pip install gym==0.21.0 numpy==1.19.5 opencv-python==4.5.2.54
  pip install "tf2onnx<1.10" "onnx<1.11"
"""

import os
import sys
import argparse
import json
import numpy as np


def find_policy_input_output(sess, model):
    """
    在 PPO2 的 TF graph 中定位 policy 的输入/输出节点。

    stable-baselines PPO2 CNN policy 内部结构:
      - 输入: observation placeholder, shape (?, 128, 128, 2) 
      - 输出: action logits (?, 10), value function (?, 1)

    只导出 policy 子图 (action logits)。
    """
    graph = sess.graph

    # 收集所有 placeholder 节点
    placeholders = []
    all_ops = graph.get_operations()
    for op in all_ops:
        if op.type == "Placeholder":
            placeholders.append(op)

    print(f"\n  Found {len(placeholders)} placeholders:")
    for p in placeholders:
        print(f"    {p.name}: {p.outputs[0].shape}")

    # 收集可能的 policy 输出
    candidates = []
    for op in all_ops:
        name = op.name.lower()
        # PPO2 CNN policy 的 action logits 通常在这些节点中
        if ("pi" in name or "policy" in name or "action" in name) and op.outputs:
            for out in op.outputs:
                shape_str = str(out.shape)
                # logits shape = (?, 10), value shape = (?, 1)
                if "10" in shape_str or out.shape.ndims == 2:
                    candidates.append((op.name, out, out.shape))

    print(f"\n  Found {len(candidates)} candidate output nodes:")
    for name, _, shape in candidates:
        print(f"    {name}: {shape}")

    # 确定输入节点 (取第一个匹配 "obs" 或通用 placeholder)
    input_name = None
    for p in placeholders:
        if "obs" in p.name.lower() or "input" in p.name.lower() or "placeholder" in p.name.lower():
            input_name = p.name
            break
    if input_name is None and placeholders:
        input_name = placeholders[0].name

    # 确定输出节点
    # 优先选择 shape 为 (?, 10) 的节点 (policy logits)
    # 排除权重/Adam 节点, 它们也是 2D 但不是 ?, 10
    # 从用户输出可以看到:
    #   model/pi/add: (?, 10)           ← 这是正确的 logits 输出
    #   train_model/model/pi/add: (?, 10)
    output_name = None
    for name, out, shape in candidates:
        try:
            ndims = shape.ndims
            if ndims is None:
                continue
            dims = shape.as_list()
            # 必须是 (?, 10) — batch dimension + 10 actions
            if dims[-1] == 10 and dims[0] is None:
                output_name = out.name
                break
        except Exception:
            pass
    if output_name is None:
        # fallback: any (?, 10) via string check
        for name, out, shape in candidates:
            shape_str = str(shape)
            if "(?, 10)" in shape_str:
                output_name = out.name
                break
    if output_name is None and candidates:
        output_name = candidates[-1][1].name

    print(f"\n  Selected: input={input_name}, output={output_name}")
    return input_name, output_name


def validate_export(sess, input_name, output_name, onnx_path, obs_shape):
    """
    对比同一 batch obs 的 TF 输出 vs ONNX 输出。
    如果 onnxruntime 未安装则跳过验证并返回 None。
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("  [SKIP] onnxruntime not installed — skipping validation")
        print("  Install: pip install onnxruntime")
        return None

    # 生成测试输入
    # PPO2 graph includes policy scale=True as input/truediv (/255).
    # Feed the raw Box(0,255) observation to both graphs; feeding obs/255 here
    # would validate the same double-normalization bug on both sides.
    obs_raw = np.random.randint(0, 256, (4, *obs_shape), dtype=np.uint8).astype(np.float32)

    # TF 推理
    tf_out = sess.run(output_name, feed_dict={f"{input_name}:0": obs_raw})

    # ONNX 推理
    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = ort_session.get_inputs()
    ort_out = ort_session.run(None, {ort_inputs[0].name: obs_raw})

    # 比较
    action_tf = np.argmax(tf_out, axis=1)
    action_onnx = np.argmax(ort_out[0], axis=1)

    print(f"\n  TF actions:    {action_tf}")
    print(f"  ONNX actions:  {action_onnx}")
    print(f"  Match:         {np.all(action_tf == action_onnx)}")
    diff = np.abs(tf_out - ort_out[0]).max()
    print(f"  Max logit diff: {diff:.6f}")

    if np.all(action_tf == action_onnx):
        print("  ✓ ALL actions match!")
    else:
        mismatch = np.where(action_tf != action_onnx)[0]
        print(f"  ✗ Mismatched samples: {mismatch}")

    return np.all(action_tf == action_onnx)


def main():
    parser = argparse.ArgumentParser(description="Export PPO2 policy to ONNX")
    parser.add_argument("--model", default="C:\\data_eve\\FaultyYawLanding\\DeepRL\\data\\trained_policy\\last_step_model.zip")
    parser.add_argument("--output", default="C:\\data_eve\\FaultyYawLanding\\DeepRL\\data\\trained_policy\\weights\\ppo2_policy.onnx")
    parser.add_argument("--meta", default=None, help="Meta JSON path (default: output+.meta.json)")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version")
    parser.add_argument("--batch-size", type=int, default=32, help="Test batch size")
    args = parser.parse_args()

    if args.meta is None:
        args.meta = args.output.replace(".onnx", "_meta.json")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # ================================================================
    # Step 1: 加载 PPO2 模型
    # ================================================================
    print("=" * 60)
    print("Step 1: Loading PPO2 model...")
    print("=" * 60)

    try:
        import tensorflow as tf
        from stable_baselines.ppo2 import PPO2
    except ImportError as e:
        print(f"[FATAL] Need TF1 + stable-baselines: {e}")
        print("  pip install tensorflow==1.15.0 stable-baselines==2.10.2")
        sys.exit(1)

    model = PPO2.load(args.model)
    sess = model.sess  # type: tf.Session
    obs_space = model.observation_space

    # 确定观测形状
    obs_shape = tuple(obs_space.shape)  # e.g., (128, 128, 2)
    print(f"  PPO2 loaded. obs_space={obs_space}, action_space={model.action_space}")
    print(f"  obs_shape={obs_shape}")

    # ================================================================
    # Step 2: 定位 policy 输入/输出节点
    # ================================================================
    print("\n" + "=" * 60)
    print("Step 2: Locating policy input/output nodes...")
    print("=" * 60)

    input_name, output_name = find_policy_input_output(sess, model)

    if input_name is None or output_name is None:
        print("[FATAL] Could not identify policy nodes. Dumping all ops for debug:")
        for op in sess.graph.get_operations():
            print(f"  {op.name} ({op.type})")
        sys.exit(1)

    # ================================================================
    # Step 3: Freeze graph (Variable → Constant) 以消除 float32_ref
    # ================================================================
    print("\n" + "=" * 60)
    print("Step 3: Freezing TF graph (Variable → Constant)...")
    print("=" * 60)

    from tensorflow.python.framework import graph_util

    output_node_name = output_name.split(":")[0]  # strip ":0"
    frozen_graph_def = graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        [output_node_name],
    )

    print(f"  Frozen graph created. Output node: {output_node_name}")

    # ================================================================
    # Step 4: tf2onnx 转换 (从 frozen graph def)
    # ================================================================
    print("\n" + "=" * 60)
    print("Step 4: Converting frozen graph to ONNX via tf2onnx...")
    print("=" * 60)

    try:
        import tf2onnx
    except ImportError:
        print("[FATAL] tf2onnx not installed: pip install tf2onnx")
        sys.exit(1)

    input_nodes = [f"{input_name}:0"]
    output_nodes = [output_name]

    print(f"  TF input:  {input_nodes}")
    print(f"  TF output: {output_nodes}")
    print(f"  Opset:     {args.opset}")

    with tf.Graph().as_default() as frozen_graph:
        tf.import_graph_def(frozen_graph_def, name="")

    onnx_graph = tf2onnx.tfonnx.process_tf_graph(
        frozen_graph,
        input_names=input_nodes,
        output_names=output_nodes,
        opset=args.opset,
    )

    model_proto = onnx_graph.make_model("ppo2_policy")

    import onnx
    onnx.save(model_proto, args.output)
    print(f"  ONNX model saved to {args.output}")

    # ================================================================
    # Step 5: 验证 TF vs ONNX 一致性
    # ================================================================
    print("\n" + "=" * 60)
    print("Step 5: Validating TF vs ONNX...")
    print("=" * 60)

    match = validate_export(sess, input_name, output_name, args.output, obs_shape)

    # ================================================================
    # Step 6: 保存元数据
    # ================================================================
    print("\n" + "=" * 60)
    print("Step 6: Saving metadata...")
    print("=" * 60)

    import onnxruntime as ort
    ort_session = ort.InferenceSession(args.output)
    ort_input = ort_session.get_inputs()[0]
    ort_output = ort_session.get_outputs()[0]

    meta = {
        "source_model": args.model,
        "input_name": input_name,
        "output_name": output_name,
        "onnx_input_name": ort_input.name,
        "onnx_input_shape": [dim if isinstance(dim, int) else -1 for dim in ort_input.shape],
        "onnx_output_name": ort_output.name,
        "onnx_output_shape": [dim if isinstance(dim, int) else -1 for dim in ort_output.shape],
        "observation_space_shape": list(obs_shape),
        "observation_space_low": float(obs_space.low.min()) if hasattr(obs_space, 'low') else 0,
        "observation_space_high": float(obs_space.high.max()) if hasattr(obs_space, 'high') else 255,
        "action_space_n": int(model.action_space.n),
        "opset_version": args.opset,
        "normalization": "ONNX graph input/truediv performs obs/255; caller feeds raw Box(0,255)",
        "export_notes": [
            "PPO2 CNN policy inference subgraph only",
            "External input: (N, H, W, C) float32, raw range [0, 255]",
            "Output: (N, 10) float32 action logits",
            "Use argmax(logits) to get discrete action 0-9",
        ],
    }
    with open(args.meta, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Meta saved to {args.meta}")

    # ================================================================
    # 总结
    # ================================================================
    print("\n" + "=" * 60)
    if match is None:
        print("✓ ONNX model saved (validation skipped — install onnxruntime to verify)")
    elif match:
        print("✓ SUCCESS — ONNX export verified!")
    else:
        print("⚠ EXPORT DONE but actions mismatch — review before Orin deployment")
    print("=" * 60)
    print(f"\nCopy to Orin:")
    print(f"  {args.output}")
    print(f"  {args.meta}")
    print(f"\nOrin inference:")
    print(f"  pip install numpy onnxruntime pyzmq opencv-python")
    print(f"  python control/drl_control_service_onnx.py \\")
    print(f"      --onnx-model {args.output} \\")
    print(f"      --onnx-meta {args.meta}")

    return 0 if match is not False else 1


if __name__ == "__main__":
    sys.exit(main())
