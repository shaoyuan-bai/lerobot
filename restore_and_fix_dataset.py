#!/usr/bin/env python3
"""检查备份并修复数据集中 action 的夹爪值"""

import pyarrow.parquet as pq
import pyarrow as pa
import json
from pathlib import Path
import shutil

dataset_path = Path(r"C:\Users\ROG\.cache\huggingface\lerobot\joyandai\lerobot_v3_rightv3")
data_file = dataset_path / "data" / "chunk-000" / "file-000.parquet"
backup_file = data_file.with_suffix('.parquet.backup')
backup_file2 = data_file.with_suffix('.parquet.backup2')

print("=== 检查数据集状态 ===\n")

# 检查备份
if backup_file2.exists():
    print(f"✅ 找到备份2: {backup_file2}")
    use_backup = backup_file2
elif backup_file.exists():
    print(f"✅ 找到备份1: {backup_file}")
    use_backup = backup_file
else:
    print(f"⚠️ 没有找到备份文件")
    use_backup = None

# 检查当前数据
if data_file.exists():
    print(f"✅ 当前数据文件存在: {data_file}")
    table = pq.read_table(data_file)
    print(f"  - 总行数: {len(table)}")
    
    # 检查是否有 episode 76
    episodes = table['episode_index'].to_pylist()
    has_ep76 = 76 in episodes
    print(f"  - 包含 Episode 76: {has_ep76}")
    
    if has_ep76:
        ep76_count = episodes.count(76)
        print(f"  - Episode 76 行数: {ep76_count}")
    
    # 检查 action 中的 gripper.pos
    action_col = table['action'].to_pylist()
    none_count = sum(1 for action in action_col if action[-1] is None)
    print(f"  - Action 中 gripper.pos 为 None 的数量: {none_count}/{len(action_col)}")
    
    if none_count > 0:
        print(f"\n⚠️ 需要修复 {none_count} 条 action 数据")
        print("\n选项：")
        if use_backup:
            print(f"1. 从备份恢复: {use_backup.name}")
            print(f"2. 修复当前数据（用 observation.state 替换 None）")
        else:
            print(f"1. 修复当前数据（用 observation.state 替换 None）")
    else:
        print(f"\n✅ 数据完好，无需修复")
else:
    print(f"❌ 数据文件不存在！")
    if use_backup:
        print(f"\n可以从备份恢复: {use_backup}")

print("\n=== 检查完成 ===")
