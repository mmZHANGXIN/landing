"""numpy-only 最近有效单元填充 (不依赖 cv2/torch).

供 perception/training_camera_projection.py 与 scripts/replay_compare_common.py
共用; 保持本模块零重依赖, 使 compare 脚本的纯 numpy 逻辑可在无 cv2/torch
的环境独立测试.
"""

from __future__ import annotations

import numpy as np


def fill_valid_nearest(values, valid, labels, chunk=2048):
    """按 cv2 距离变换标签把无效单元填成其 label 组内 L2 最近有效单元的 value.

    cv2.distanceTransformWithLabels 的 label 语义因版本而异:
    - cv2 < 5.0: DIST_LABEL_PIXEL = 每个 zero(有效)像素独立 label (文档语义),
      label 即有效单元的紧凑序号, ``valid_coords[label-1]`` 即为最近单元.
    - cv2 >= 5.0: PIXEL ≡ CCOMP, 按 8-连通分量打 label (5.0.0 实测, 相邻 zero
      同 label), ``valid_coords[label-1]`` 会错选分量内栅格序第一个单元.

    本函数对两种语义都给出正确结果: 同一 label 的单元构成候选集合, 无效单元
    取其集合内 L2 最近者. 4.x 下输出与紧凑索引法逐位一致; 5.x 下修正分量
    语义偏差. 按 chunk 分批计算距离矩阵, 避免超大分量时内存溢出.

    调用方必须把距离变换输入设为 ``(~valid).astype(np.uint8)`` (零像素 = 真实
    有效单元), 否则 label 会打在空洞上, 填充结果错误.
    """
    filled = values.copy()
    if not valid.any():
        return filled
    hole_pts = np.argwhere(~valid)
    if hole_pts.size == 0:
        return filled
    valid_pts = np.argwhere(valid)
    vlab = labels[valid_pts[:, 0], valid_pts[:, 1]]
    for l in np.unique(vlab):
        cand = valid_pts[vlab == l]
        sel = hole_pts[labels[hole_pts[:, 0], hole_pts[:, 1]] == l]
        if len(sel) == 0:
            continue
        for s0 in range(0, len(sel), chunk):
            d2 = np.sum(
                (sel[s0:s0 + chunk, None, :] - cand[None, :, :]) ** 2, axis=-1)
            j = np.argmin(d2, axis=1)
            filled[sel[s0:s0 + chunk, 0], sel[s0:s0 + chunk, 1]] = \
                values[cand[j, 0], cand[j, 1]]
    return filled
