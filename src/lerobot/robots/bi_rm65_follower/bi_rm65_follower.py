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

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs

from ..robot import Robot
from .config_bi_rm65_follower import BiRM65FollowerConfig, RM65FollowerConfig
from .rm65_follower import RM65Follower

logger = logging.getLogger(__name__)


class BiRM65Follower(Robot):
    """
    睿尔曼 RM65 双臂机器人系统
    
    控制两台 RM65 机械臂组成的双臂系统，通过网络分别连接左右两臂。
    适用于需要双臂协同操作的任务场景。
    """

    config_class = BiRM65FollowerConfig
    name = "bi_rm65_follower"

    def __init__(self, config: BiRM65FollowerConfig):
        super().__init__(config)
        self.config = config

        # 创建左臂配置
        left_arm_config = RM65FollowerConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            ip_address=config.left_arm_ip,
            port=config.port,
            move_speed=config.move_speed,
            cameras={},
        )

        # 创建右臂配置
        right_arm_config = RM65FollowerConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            ip_address=config.right_arm_ip,
            port=config.port,
            move_speed=config.move_speed,
            cameras={},
        )

        # 初始化两个机械臂
        self.left_arm = RM65Follower(left_arm_config)
        self.right_arm = RM65Follower(right_arm_config)

        # 共享相机系统
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        """定义所有关节的特征 (左臂 + 右臂)"""
        return {f"left_{motor}.pos": float for motor in self.left_arm.joint_names} | {
            f"right_{motor}.pos": float for motor in self.right_arm.joint_names
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """定义相机特征"""
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """组合电机和相机的观察特征"""
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """动作特征与电机特征相同"""
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        """检查两个机械臂和所有相机是否都已连接"""
        return (
            self.left_arm.is_connected
            and self.right_arm.is_connected
            and all(cam.is_connected for cam in self.cameras.values())
        )

    def connect(self, calibrate: bool = True) -> None:
        """连接两个机械臂和所有相机"""
        logger.info("Connecting left arm...")
        self.left_arm.connect(calibrate)

        logger.info("Connecting right arm...")
        self.right_arm.connect(calibrate)

        logger.info("Connecting cameras...")
        for cam in self.cameras.values():
            cam.connect()

        logger.info(f"{self} connected successfully.")

    @property
    def is_calibrated(self) -> bool:
        """两个机械臂都校准才算校准完成"""
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        """校准两个机械臂"""
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        """配置两个机械臂"""
        self.left_arm.configure()
        self.right_arm.configure()

    def get_observation(self) -> dict[str, Any]:
        """获取双臂观察数据"""
        obs_dict = {}

        # 获取左臂观察 (添加 "left_" 前缀)
        left_obs = self.left_arm.get_observation()
        obs_dict.update({f"left_{key}": value for key, value in left_obs.items()})

        # 获取右臂观察 (添加 "right_" 前缀)
        right_obs = self.right_arm.get_observation()
        obs_dict.update({f"right_{key}": value for key, value in right_obs.items()})

        # 获取相机图像
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """发送动作到两个机械臂"""
        # 分离左右臂动作
        left_action = {
            key.removeprefix("left_"): value for key, value in action.items() if key.startswith("left_")
        }
        right_action = {
            key.removeprefix("right_"): value for key, value in action.items() if key.startswith("right_")
        }

        # 并行发送到两个机械臂
        send_action_left = self.left_arm.send_action(left_action)
        send_action_right = self.right_arm.send_action(right_action)

        # 添加前缀并返回
        return {
            **{f"left_{key}": value for key, value in send_action_left.items()},
            **{f"right_{key}": value for key, value in send_action_right.items()},
        }

    def disconnect(self):
        """断开两个机械臂和所有相机"""
        logger.info("Disconnecting left arm...")
        self.left_arm.disconnect()

        logger.info("Disconnecting right arm...")
        self.right_arm.disconnect()

        logger.info("Disconnecting cameras...")
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
