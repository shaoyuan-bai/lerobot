"""
修复数据集中 action 的 gripper.pos None 值
将其替换为对应 observation.state 中的夹爪值
"""
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import shutil

# 数据集路径
dataset_path = Path(r"C:\Users\ROG\.cache\huggingface\lerobot\joyandai\lerobot_v3_rightv3")
data_dir = dataset_path / "data"

print(f"正在处理数据集: {dataset_path}")

# 遍历所有 parquet 文件
parquet_files = list(data_dir.rglob("*.parquet"))
print(f"找到 {len(parquet_files)} 个 parquet 文件")

for file_path in parquet_files:
    print(f"\n处理: {file_path}")
    
    # 读取原始数据
    table = pq.read_table(file_path)
    
    # 提取列
    action_col = table.column('action').to_pylist()
    obs_state_col = table.column('observation.state').to_pylist()
    
    # 统计修复数量
    fixed_count = 0
    
    # 修复 action 中的 None
    for i in range(len(action_col)):
        if action_col[i][-1] is None:  # 夹爪位置是 None
            # 用 observation.state 的夹爪值替换
            action_col[i][-1] = obs_state_col[i][-1]
            fixed_count += 1
    
    print(f"  修复了 {fixed_count} 条记录")
    
    # 备份原文件
    backup_path = file_path.with_suffix('.parquet.backup')
    if not backup_path.exists():
        shutil.copy2(file_path, backup_path)
        print(f"  已备份到: {backup_path}")
    
    # 重建 action 列
    action_array = pa.array(action_col, type=pa.list_(pa.float32(), 7))
    
    # 替换表中的 action 列
    action_field_index = table.schema.get_field_index('action')
    new_table = table.set_column(action_field_index, 'action', action_array)
    
    # 写回文件
    pq.write_table(new_table, file_path)
    print(f"  ✓ 已保存")

print("\n=== 修复完成 ===")
print("如需恢复，删除 .parquet 文件并将 .parquet.backup 重命名回 .parquet")
