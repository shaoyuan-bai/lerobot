#!/usr/bin/env python
"""
将自定义RM65录制格式转换为LeRobot v3.0标准格式

自定义格式:
outputs/rm65_recordings/
├── dataset_summary.json
├── episode_0000/
│   ├── metadata.json
│   ├── states.json
│   └── images/
│       └── top_*.jpg

LeRobot v3.0格式:
data/
├── chunk-000/
│   └── file-000.parquet
├── meta/
│   ├── info.json
│   ├── stats.json
│   └── episodes/
│       └── chunk-000/
│           └── file-000.parquet
└── videos/
    └── observation.images.top/
        └── chunk-000/
            └── file-000.mp4

使用方法:
python convert_custom_to_lerobot.py \
  --input_dir outputs/rm65_recordings \
  --repo_id woosh/rm65_converted \
  --fps 30
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame


def load_episode_data(episode_dir: Path):
    """加载单个episode的数据"""
    # 读取metadata
    with open(episode_dir / "metadata.json") as f:
        metadata = json.load(f)
    
    # 读取states
    with open(episode_dir / "states.json") as f:
        states = json.load(f)
    
    # 读取图像
    images_dir = episode_dir / "images"
    image_files = sorted(images_dir.glob("top_*.jpg"))
    
    return metadata, states, image_files


def convert_state_to_observation(state_dict):
    """将state字典转换为observation格式"""
    # state_dict格式: {"left_joint_1.pos": 13.7, ...}
    observation = {}
    for key, value in state_dict.items():
        # 转换为tensor
        observation[key] = torch.tensor([value], dtype=torch.float32)
    return observation


def convert_dataset(input_dir: Path, repo_id: str, output_dir: Path = None, fps: int = 30):
    """转换整个数据集"""
    print(f"\n{'='*60}")
    print(f"转换数据集: {input_dir}")
    print(f"目标格式: LeRobot v3.0")
    print(f"{'='*60}\n")
    
    # 读取数据集摘要
    with open(input_dir / "dataset_summary.json") as f:
        summary = json.load(f)
    
    print(f"找到 {summary['num_episodes']} 个episodes")
    
    # 直接使用hw_to_dataset_features构建features
    print("\n正在创建LeRobot数据集结构...")
    from lerobot.datasets.utils import hw_to_dataset_features, combine_feature_dicts
    
    # 定义机器人硬件特征
    obs_hw_features = {
        "left_joint_1.pos": float,
        "left_joint_2.pos": float,
        "left_joint_3.pos": float,
        "left_joint_4.pos": float,
        "left_joint_5.pos": float,
        "left_joint_6.pos": float,
        "right_joint_1.pos": float,
        "right_joint_2.pos": float,
        "right_joint_3.pos": float,
        "right_joint_4.pos": float,
        "right_joint_5.pos": float,
        "right_joint_6.pos": float,
        "top": (480, 640, 3),  # 相机图像
    }
    
    action_hw_features = {
        "left_joint_1.pos": float,
        "left_joint_2.pos": float,
        "left_joint_3.pos": float,
        "left_joint_4.pos": float,
        "left_joint_5.pos": float,
        "left_joint_6.pos": float,
        "right_joint_1.pos": float,
        "right_joint_2.pos": float,
        "right_joint_3.pos": float,
        "right_joint_4.pos": float,
        "right_joint_5.pos": float,
        "right_joint_6.pos": float,
    }
    
    # 转换为数据集格式
    features = combine_feature_dicts(
        hw_to_dataset_features(obs_hw_features, "observation", use_video=True),
        hw_to_dataset_features(action_hw_features, "action", use_video=True),
    )
    
    # 创建LeRobot数据集
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=output_dir,
        robot_type="bi_rm65_follower",
        features=features,
        use_videos=True,
        image_writer_threads=4,
    )
    
    # 启动图像写入器
    dataset.start_image_writer(num_processes=0, num_threads=4)
    
    print(f"✓ 数据集已创建: {dataset.root}\n")
    
    # 转换每个episode
    episode_dirs = sorted(input_dir.glob("episode_*"))
    
    for episode_dir in tqdm(episode_dirs, desc="转换episodes"):
        episode_id = int(episode_dir.name.split("_")[1])
        
        print(f"\n转换 Episode {episode_id}...")
        metadata, states, image_files = load_episode_data(episode_dir)
        
        # 确保数据一致
        assert len(states) == len(image_files), \
            f"Episode {episode_id}: states({len(states)}) != images({len(image_files)})"
        
        # 逐帧添加
        for i, (state, img_path) in enumerate(zip(states, image_files)):
            # 读取图像
            img = Image.open(img_path)
            img_array = np.array(img)
            
            # 构建observation
            observation = convert_state_to_observation(state)
            observation["images.top"] = torch.from_numpy(img_array)
            
            # action = observation (Follower模式)
            action = {k: v for k, v in observation.items() if not k.startswith("images.")}
            
            # 构建frame
            frame = build_dataset_frame(
                observation=observation,
                action=action,
            )
            
            # 添加到数据集
            dataset.add_frame(frame)
        
        # 保存episode
        dataset.save_episode(task="rm65_demo")
        print(f"  ✓ Episode {episode_id}: {len(states)} frames")
    
    # 停止图像写入器并合并数据
    print("\n正在合并数据集...")
    dataset.stop_image_writer()
    dataset.consolidate()
    
    print(f"\n{'='*60}")
    print(f"✓ 转换完成!")
    print(f"输出路径: {dataset.root}")
    print(f"{'='*60}")
    print("\n数据集结构:")
    print("  ├── data/chunk-000/file-000.parquet")
    print("  ├── meta/info.json, stats.json")
    print("  └── videos/observation.images.top/chunk-000/file-000.mp4")
    
    return dataset


def main():
    parser = argparse.ArgumentParser(description="转换自定义格式到LeRobot v3.0")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="自定义格式数据集路径 (例如: outputs/rm65_recordings)"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="LeRobot数据集ID (例如: woosh/rm65_converted)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出路径 (默认: ~/.cache/huggingface/lerobot/{repo_id})"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="帧率 (默认: 30)"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    if not input_dir.exists():
        print(f"错误: 输入目录不存在: {input_dir}")
        return
    
    if not (input_dir / "dataset_summary.json").exists():
        print(f"错误: 未找到dataset_summary.json,这不是有效的自定义格式数据集")
        return
    
    convert_dataset(input_dir, args.repo_id, output_dir, args.fps)


if __name__ == "__main__":
    main()
