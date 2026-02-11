#!/usr/bin/env python3
"""从 HuggingFace Hub 恢复数据集"""

from huggingface_hub import snapshot_download
import time

print("正在从 HuggingFace Hub 重新下载数据集...")
print("数据集: joyandai/lerobot_v3_rightv3")
print()

start_time = time.time()

try:
    path = snapshot_download(
        repo_id='joyandai/lerobot_v3_rightv3',
        repo_type='dataset'
    )
    
    elapsed = time.time() - start_time
    print(f"\n✅ 数据集已成功恢复！")
    print(f"📁 路径: {path}")
    print(f"⏱️  耗时: {elapsed:.1f} 秒")
    
except Exception as e:
    print(f"\n❌ 下载失败: {e}")
    print("\n可能的解决方案:")
    print("1. 检查网络连接")
    print("2. 确认 HuggingFace token 配置正确")
    print("3. 手动运行: huggingface-cli download joyandai/lerobot_v3_rightv3 --repo-type dataset")
