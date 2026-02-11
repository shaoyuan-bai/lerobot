import pyarrow.parquet as pq
import sys
import io
from collections import Counter

# 修复 Windows 中文编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 读取数据
table = pq.read_table(r'C:\Users\ROG\.cache\huggingface\lerobot\joyandai\lerobot_v3_rightv3\data\chunk-000\file-000.parquet')

print("=== 统计 action 中的夹爪位置分布 ===")
action_col = table.column('action')
gripper_values = [action_col[i].as_py()[-1] for i in range(len(action_col))]

# 统计不同值
counter = Counter(gripper_values)
print(f"\n总样本数: {len(gripper_values)}")
print(f"唯一值数量: {len(counter)}")

# 显示所有唯一值及其数量
print("\n夹爪位置值分布:")
for value, count in sorted(counter.items(), key=lambda x: x[1], reverse=True)[:20]:
    percentage = count / len(gripper_values) * 100
    print(f"  {value:.2f}: {count} 次 ({percentage:.2f}%)")

# 检查是否真的只有两个值
print(f"\n=== 统计 observation.state 中的夹爪位置分布 ===")
obs_col = table.column('observation.state')
obs_gripper_values = [obs_col[i].as_py()[-1] for i in range(len(obs_col))]

obs_counter = Counter(obs_gripper_values)
print(f"\n总样本数: {len(obs_gripper_values)}")
print(f"唯一值数量: {len(obs_counter)}")

print("\n夹爪位置值分布:")
for value, count in sorted(obs_counter.items(), key=lambda x: x[1], reverse=True)[:20]:
    percentage = count / len(obs_gripper_values) * 100
    print(f"  {value:.2f}: {count} 次 ({percentage:.2f}%)")
