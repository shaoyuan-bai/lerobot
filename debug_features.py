#!/usr/bin/env python
"""调试特征配置"""

from lerobot.robots.bi_rm65_follower import BiRM65FollowerConfig, BiRM65Follower
from lerobot.cameras.opencv import OpenCVCameraConfig

# 相机配置
cameras_config = {
    "top": OpenCVCameraConfig(
        index_or_path=0,
        fps=30,
        width=640,
        height=480,
    ),
}

# 机器人配置
config = BiRM65FollowerConfig(
    id="rm65_recorder",
    left_arm_ip="169.254.128.20",
    right_arm_ip="169.254.128.21",
    port=8080,
    move_speed=30,
    enable_right_gripper=True,  # 启用右臂夹爪
    gripper_device_id=9,
    gripper_force=60,
    gripper_speed=255,
    cameras=cameras_config,
)

robot = BiRM65Follower(config)

print("=" * 60)
print("机器人特征配置")
print("=" * 60)

print("\n观察特征 (observation_features):")
for key, value in robot.observation_features.items():
    print(f"  {key}: {value}")

print(f"\n总数: {len(robot.observation_features)}")

print("\n动作特征 (action_features):")
for key, value in robot.action_features.items():
    print(f"  {key}: {value}")

print(f"\n总数: {len(robot.action_features)}")

print("\n配置信息:")
print(f"  enable_right_gripper: {config.enable_right_gripper}")
print(f"  gripper_device_id: {config.gripper_device_id}")
