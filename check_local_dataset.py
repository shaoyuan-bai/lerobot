#!/usr/bin/env python3
"""恢复本地数据集并删除 episode 76"""

import os
from pathlib import Path
import shutil

# 检查回收站或备份
dataset_path = Path(r"C:\Users\ROG\.cache\huggingface\lerobot\joyandai\lerobot_v3_rightv3")
parent_path = dataset_path.parent

print(f"=== 检查数据集路径 ===")
print(f"目标路径: {dataset_path}")
print(f"是否存在: {dataset_path.exists()}")

if not dataset_path.exists():
    print(f"\n❌ 数据集已被删除！")
    print(f"\n正在检查父目录...")
    if parent_path.exists():
        print(f"✅ 父目录存在: {parent_path}")
        subdirs = list(parent_path.iterdir())
        print(f"子目录列表: {[d.name for d in subdirs]}")
    else:
        print(f"❌ 父目录也不存在: {parent_path}")
    
    print(f"\n可能的解决方案：")
    print(f"1. 从 Windows 回收站恢复")
    print(f"2. 从备份恢复")
    print(f"3. 从 HuggingFace Hub 重新下载")
    print(f"4. 从 Jetson 设备重新传输")
else:
    print(f"✅ 数据集存在，可以继续操作")
