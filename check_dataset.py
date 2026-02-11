import pyarrow.parquet as pq
import sys
import io

# 修复 Windows 中文编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 读取数据
table = pq.read_table(r'C:\Users\ROG\.cache\huggingface\lerobot\joyandai\lerobot_v3_rightv3\data\chunk-000\file-000.parquet')

print("=== 数据集列名 ===")
print(table.column_names[:15])

print("\n=== Action 前5条 ===")
action_col = table.column('action')
for i in range(min(5, len(action_col))):
    action = action_col[i].as_py()
    gripper_pos = action[-1]
    print(f"  [{i}] joints={action[:6]}, gripper={gripper_pos}")
    if gripper_pos is None:
        print(f"       [WARNING] 夹爪位置是 None!")

print("\n=== Observation.state 前5条 ===")
obs_col = table.column('observation.state')
for i in range(min(5, len(obs_col))):
    obs = obs_col[i].as_py()
    gripper_pos = obs[-1]
    print(f"  [{i}] joints={obs[:6]}, gripper={gripper_pos}")
    if gripper_pos is None:
        print(f"       [WARNING] 夹爪位置是 None!")

print("\n=== 统计 ===")
none_count_action = sum(1 for i in range(len(action_col)) if action_col[i].as_py()[-1] is None)
none_count_obs = sum(1 for i in range(len(obs_col)) if obs_col[i].as_py()[-1] is None)
print(f"Action中夹爪为None的数量: {none_count_action}/{len(action_col)}")
print(f"Observation.state中夹爪为None的数量: {none_count_obs}/{len(obs_col)}")
