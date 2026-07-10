#!/usr/bin/env python3
"""
rosbagconvert.py — 将 ROS2 rosbag (sqlite3/db3) 转换为 ROS1 .bag 格式
====================================================================
自动注册 livox_ros_driver2 自定义消息 (CustomMsg, CustomPoint),
适用于 Jetson Orin + ROS2 Galactic + livox_ws 环境。

用法:
  source /opt/ros/galactic/setup.bash
  python3 scripts/rosbagconvert.py [--input ROS2_DIR] [--output OUTPUT.bag]
"""

import argparse
from pathlib import Path

from rosbags.rosbag1 import Writer
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg


# ---- Orin 实际路径: livox_ros_driver2 消息定义目录 ----
LIVOX_MSG_DIRS = [
    Path.home() / "livox_ws/src/livox_ros_driver2/msg",
    Path.home() / "livox_ws/install/livox_ros_driver2/share/livox_ros_driver2/msg",
]


def _find_livox_msg_dir() -> Path:
    """自动查找 livox_ros_driver2 消息定义目录."""
    for d in LIVOX_MSG_DIRS:
        if d.is_dir() and (d / "CustomMsg.msg").exists():
            return d
    raise FileNotFoundError(
        f"Cannot find livox_ros_driver2/msg. Searched: {LIVOX_MSG_DIRS}"
    )


def register_custom_types(typestore, msg_dir: Path) -> None:
    """将 livox_ros_driver2 的自定义消息注册到类型存储."""
    for msg_file in sorted(msg_dir.glob("*.msg")):
        type_name = f"livox_ros_driver2/msg/{msg_file.stem}"
        typestore.register(get_types_from_msg(msg_file.read_text(), type_name))
        print(f"  [registered] {type_name}")


def main():
    parser = argparse.ArgumentParser(
        description="ROS2 rosbag → ROS1 .bag 转换 (Orin/Galactic)"
    )
    parser.add_argument(
        "--input", "-i",
        default="/home/orin/evelyn/landing/rosbag2_2026_06_10-17_52_00",
        help="ROS2 rosbag 目录路径",
    )
    parser.add_argument(
        "--output", "-o",
        help="输出 .bag 文件路径 (默认: <input>/rosbag_out.bag)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_dir():
        print(f"ERROR: input directory not found: {input_path}")
        return 1

    output_path = Path(args.output) if args.output else (input_path / "rosbag_out.bag")

    # 1. 查找并注册 livox 自定义消息
    msg_dir = _find_livox_msg_dir()
    print(f"Using msg dir: {msg_dir}")

    # 2. 创建 ROS2 Galactic 类型存储
    typestore = get_typestore(Stores.ROS2_GALACTIC)
    register_custom_types(typestore, msg_dir)

    # 3. 转换
    print(f"Converting: {input_path} → {output_path}")
    with Reader(str(input_path)) as reader, Writer(str(output_path)) as writer:
        writer_conns = {}  # topic → writer connection

        for conn, timestamp, raw_data in reader.messages():
            topic = conn.topic
            msgtype_str = conn.msgtype

            # 为每个 topic 创建一次 writer connection
            if topic not in writer_conns:
                try:
                    wconn = writer.add_connection(topic, msgtype_str, typestore=typestore)
                    writer_conns[topic] = wconn
                    print(f"  [topic] {topic} ({msgtype_str})")
                except Exception as exc:
                    print(f"  [skip] {topic} ({msgtype_str}) — {exc}")
                    continue

            # CDR → Python → ROS1
            try:
                msg = typestore.deserialize_cdr(raw_data, msgtype_str)
                ros1_raw = typestore.serialize_ros1(msg, msgtype_str)
                writer.write(writer_conns[topic], timestamp, ros1_raw)
            except Exception as exc:
                print(f"  [warn] {topic} @ {timestamp}: {exc}")
                continue

    print(f"Done → {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())