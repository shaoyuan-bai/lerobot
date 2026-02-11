#!/usr/bin/env python3
"""检查 episode 76 是否还存在"""

import pyarrow.parquet as pq
import json
from pathlib import Path

# 数据集路径
dataset_path = Path.home() / ".cache/huggingface/lerobot/joyandai/lerobot_v3_rightv3"

print(f"=== 检查数据集: {dataset_path} ===\n")

# 检查路径是否存在
if not dataset_path.exists():
    print(f"❌ 数据集路径不存在！")
    print("可能的原因：")
    print("1. 数据集已经被删除")
    print("2. 数据集在 HuggingFace Hub 上，需要先下载")
    print("\n尝试查找其他可能的位置...")
    
    # 检查 datasets 缓存
    datasets_cache = Path.home() / ".cache/huggingface/datasets"
    if datasets_cache.exists():
        print(f"\n找到 datasets 缓存: {datasets_cache}")
        subdirs = list(datasets_cache.glob("*lerobot*"))
        for d in subdirs:
            print(f"  - {d.name}")
    exit(0)

# 检查 parquet 文件
parquet_file = dataset_path / "file-000.parquet"
if not parquet_file.exists():
    print(f"❌ Parquet 文件不存在: {parquet_file}")
    exit(1)

# 读取数据
print(f"✅ 读取 {parquet_file}")
table = pq.read_table(parquet_file)

# 统计 episode
episodes = table['episode_index'].to_pylist()
min_ep = min(episodes)
max_ep = max(episodes)
total_rows = len(table)

print(f"\n📊 数据统计:")
print(f"  - Episode 范围: {min_ep} - {max_ep}")
print(f"  - 总行数: {total_rows}")

# 检查 episode 76
import collections
counts = collections.Counter(episodes)
ep76_count = counts.get(76, 0)

if ep76_count > 0:
    print(f"\n⚠️ Episode 76 仍然存在！")
    print(f"  - Episode 76 的行数: {ep76_count}")
    print("\n建议：重新运行删除脚本")
else:
    print(f"\n✅ Episode 76 已成功删除！")

# 检查元数据
info_file = dataset_path / "info.json"
if info_file.exists():
    with open(info_file) as f:
        info = json.load(f)
    print(f"\n📝 元数据信息:")
    print(f"  - total_episodes: {info.get('total_episodes', 'N/A')}")
    print(f"  - total_frames: {info.get('total_frames', 'N/A')}")
    
    # 验证一致性
    if info.get('total_frames') == total_rows:
        print(f"  ✅ total_frames 与实际行数一致")
    else:
        print(f"  ⚠️ total_frames ({info.get('total_frames')}) 与实际行数 ({total_rows}) 不一致")

print("\n=== 检查完成 ===")
