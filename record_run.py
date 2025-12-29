import subprocess
import datetime

# 基本配置
HF_USER = "joyandai"
ROBOT_ID = "R12252801"
ROBOT_PORT = "COM3"      # Follower
LEADER_ID = "R07252801"
LEADER_PORT = "COM7"     # Leader
TASK_NAME = "Grab the black cube"

# 相机配置（写成一行，避免 Windows 解析错误）
cameras = "{'fixed':{'type':'opencv','index_or_path':1,'width':640,'height':360,'fps':30},'handeye':{'type':'opencv','index_or_path':2,'width':640,'height':360,'fps':30},'aux':{'type':'opencv','index_or_path':0,'width':640,'height':360,'fps':30}}"

# 自动生成时间戳
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
repo_id = f"{HF_USER}/so101_test_{timestamp}"

cmd = [
    "python", "-m", "lerobot.record",
    "--robot.disable_torque_on_disconnect=true",
    "--robot.type=so101_follower",
    f"--robot.port={ROBOT_PORT}",
    f"--robot.id={ROBOT_ID}",
    f"--robot.cameras={cameras}",
    "--teleop.type=so101_leader",
    f"--teleop.port={LEADER_PORT}",
    f"--teleop.id={LEADER_ID}",
    "--display_data=true",
    f"--dataset.repo_id={repo_id}",
    "--dataset.num_episodes=10",
    "--dataset.episode_time_s=20",
    f"--dataset.single_task={TASK_NAME}"
]

print("Running:", " ".join(cmd))
subprocess.run(cmd)
