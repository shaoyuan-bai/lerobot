#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@dataclass
class RM65FollowerConfig(RobotConfig):
    """睿尔曼 RM65 单臂配置"""

    # 机械臂 IP 地址
    ip_address: str

    # 端口号，默认 8080
    port: int = 8080

    # 运动速度 (1-100)，默认 50
    move_speed: int = 50

    # 是否启用夹爪
    enable_gripper: bool = False

    # 夹爪设备ID
    gripper_device_id: int = 9

    # 夹爪力度 (0-255)
    gripper_force: int = 60

    # 夹爪速度 (0-255)
    gripper_speed: int = 255

    # 相机配置
    cameras: dict[str, CameraConfig] = field(default_factory=dict)


@RobotConfig.register_subclass("bi_rm65_follower")
@dataclass
class BiRM65FollowerConfig(RobotConfig):
    """睿尔曼 RM65 双臂配置"""

    # 左臂 IP 地址
    left_arm_ip: str

    # 右臂 IP 地址
    right_arm_ip: str

    # 端口号，默认 8080
    port: int = 8080

    # 运动速度 (1-100)，默认 50
    move_speed: int = 50

    # 是否启用右臂夹爪
    enable_right_gripper: bool = False

    # 夹爪设备ID
    gripper_device_id: int = 9

    # 夹爪力度 (0-255)
    gripper_force: int = 60

    # 夹爪速度 (0-255)
    gripper_speed: int = 255

    # 相机配置(两臂共享)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
