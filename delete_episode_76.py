"""
删除最后一个不完整的 episode (76)
"""
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import shutil
import json

dataset_path = Path(r"C:\Users\ROG\.cache\huggingface\lerobot\joyandai\lerobot_v3_rightv3")

print("=== 删除 Episode 76 ===\n")

# 1. 处理数据文件
data_file = dataset_path / "data" / "chunk-000" / "file-000.parquet"
print(f"处理数据文件: {data_file}")

table = pq.read_table(data_file)
episode_indices = table.column('episode_index').to_pylist()

# 统计
total_rows = len(table)
ep76_count = episode_indices.count(76)
print(f"  总行数: {total_rows}")
print(f"  Episode 76 的行数: {ep76_count}")

# 过滤掉 episode 76
mask = [ep != 76 for ep in episode_indices]
filtered_table = table.filter(pa.array(mask))

print(f"  删除后剩余: {len(filtered_table)} 行")

# 备份
backup_file = data_file.with_suffix('.parquet.backup2')
if not backup_file.exists():
    shutil.copy2(data_file, backup_file)
    print(f"  已备份到: {backup_file}")

# 保存
pq.write_table(filtered_table, data_file)
print("  ✓ 数据文件已更新")

# 2. 更新 info.json
info_file = dataset_path / "meta" / "info.json"
print(f"\n处理元数据: {info_file}")

with open(info_file, 'r') as f:
    info = json.load(f)

old_episodes = info['total_episodes']
old_frames = info['total_frames']

info['total_episodes'] = 76  # 更新为 76 (0-75)
info['total_frames'] = len(filtered_table)

print(f"  Episodes: {old_episodes} → {info['total_episodes']}")
print(f"  Frames: {old_frames} → {info['total_frames']}")

# 备份
backup_info = info_file.with_suffix('.json.backup2')
if not backup_info.exists():
    shutil.copy2(info_file, backup_info)

with open(info_file, 'w') as f:
    json.dump(info, f, indent=2)
print("  ✓ 元数据已更新")

# 3. 删除 episode 76 的视频文件
print("\n删除视频文件:")
videos_dir = dataset_path / "videos"
for video_key in ['observation.images.handeye', 'observation.images.fixed']:
    video_file = videos_dir / video_key / "chunk-000" / "episode_000076.mp4"
    if video_file.exists():
        video_file.unlink()
        print(f"  ✓ 已删除: {video_file}")

print("\n=== 完成 ===")
print("请清除 HuggingFace 缓存后重新训练:")
print('  Remove-Item -Recurse -Force "$env:USERPROFILE\\.cache\\huggingface\\datasets"')

