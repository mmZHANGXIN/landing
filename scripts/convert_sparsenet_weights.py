#!/usr/bin/env python3
"""
SparseNet TF checkpoint → PyTorch .pth 权重转换
================================================
在任意有 TF 的机器上运行，然后 scp 到 Jetson。

用法:
  pip install tensorflow-cpu torch numpy
  python convert_sparsenet_weights.py \
    --tf_ckpt <path>/20200522-144740.ckpt \
    --out sparsenet.pth

验证 (Jetson):
  python -c "
  import sys; sys.path.insert(0,'/home/orin/evelyn/orin_landing')
  from perception.sparse_depth_completion import DepthCompletion
  dc = DepthCompletion({'dmax':30,'weight_path':'weights/sparsenet.pth'})
  print('PASS' if dc.weights_ready else 'FAIL')
  "
"""
import os, sys, argparse, numpy as np

def extract_and_convert(tf_ckpt_path, pt_out_path):
    import tensorflow as tf, torch

    reader = tf.compat.v1.train.NewCheckpointReader(tf_ckpt_path)
    all_keys = sorted(reader.get_variable_to_shape_map().keys())

    print("=== TF Keys ===")
    for k in all_keys:
        print(f"  {k}  shape={list(reader.get_tensor(k).shape)}")

    conv_keys = sorted([k for k in all_keys if 'kernel' in k.lower()])
    bias_keys = sorted([k for k in all_keys if 'variable' in k.lower() or 'bias' in k.lower()])

    if len(conv_keys) != 6:
        print(f"ERROR: Expected 6 conv kernels, got {len(conv_keys)}"); sys.exit(1)

    pt_state = {}
    expected = [
        (1, 16, 11, 11), (16, 16, 7, 7), (16, 16, 5, 5),
        (16, 16, 3, 3), (16, 16, 3, 3), (16, 1, 1, 1)
    ]
    expected_bias = [16, 16, 16, 16, 16, 1]

    for i, ck in enumerate(conv_keys):
        w_tf = reader.get_tensor(ck)               # TF: [H,W,in,out]
        w_pt = np.transpose(w_tf, (3,2,0,1)).copy() # PT: [out,in,H,W]
        assert w_pt.shape == expected[i], f"Layer {i}: {w_pt.shape} != {expected[i]}"
        pt_state[f"layers.{i}.conv.weight"] = torch.from_numpy(w_pt).float()
        pt_state[f"layers.{i}._norm_conv.weight"] = torch.ones(w_pt.shape[0], 1, *w_pt.shape[2:])
        if i < len(bias_keys):
            b = reader.get_tensor(bias_keys[i])
            assert b.shape == (expected_bias[i],), f"Bias {i}: {b.shape}"
            pt_state[f"layers.{i}.bias"] = torch.from_numpy(b.copy()).float()
        print(f"  Layer {i}: TF {list(w_tf.shape)} → PT {list(w_pt.shape)} ✓")

    os.makedirs(os.path.dirname(pt_out_path) or ".", exist_ok=True)
    torch.save(pt_state, pt_out_path)
    print(f"\nSaved: {pt_out_path} ({len(pt_state)} tensors, {os.path.getsize(pt_out_path)/1024:.0f}KB)")
    print("Done. scp to Jetson and set depth_completion.weight_path in config.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tf_ckpt", required=True)
    p.add_argument("--out", default="sparsenet.pth")
    args = p.parse_args()
    extract_and_convert(args.tf_ckpt, args.out)
