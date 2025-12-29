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
from functools import cached_property

from ..teleoperator import Teleoperator
from .config_bi_rm65_leader import BiRM65LeaderConfig, RM65LeaderConfig
from .rm65_leader import RM65Leader

logger = logging.getLogger(__name__)


class BiRM65Leader(Teleoperator):
    """
    睿尔曼 RM65 双臂主臂 (通过拖动示教读取关节角度)
    
    使用两个 RM65 的拖动示教功能,允许手动移动左右臂并读取关节角度。
    """

    config_class = BiRM65LeaderConfig
    name = "bi_rm65_leader"

    def __init__(self, config: BiRM65LeaderConfig):
        super().__init__(config)
        self.config = config

        # 创建左臂配置
        left_id = f"{config.id}_left" if config.id else None
        left_arm_config = RM65LeaderConfig(
            id=left_id,
            calibration_dir=config.calibration_dir,
            arm_ip=config.left_arm_ip,
            port=config.port,
            drag_sensitivity=config.drag_sensitivity,
        )

        # 创建右臂配置
        right_id = f"{config.id}_right" if config.id else None
        right_arm_config = RM65LeaderConfig(
            id=right_id,
            calibration_dir=config.calibration_dir,
            arm_ip=config.right_arm_ip,
            port=config.port,
            drag_sensitivity=config.drag_sensitivity,
        )

        self.left_arm = RM65Leader(left_arm_config)
        self.right_arm = RM65Leader(right_arm_config)

    @cached_property
    def action_features(self) -> dict[str, type]:
        """返回动作特征 (左右臂各 6 个关节)"""
        return {f"left_{motor}.pos": float for motor in self.left_arm.joint_names} | {
            f"right_{motor}.pos": float for motor in self.right_arm.joint_names
        }

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        """RM65 不支持力反馈"""
        return {}

    @property
    def is_connected(self) -> bool:
        """检查左右臂是否都已连接"""
        return self.left_arm.is_connected and self.right_arm.is_connected

    def connect(self, calibrate: bool = True) -> None:
        """连接左右臂并启动拖动示教"""
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

    @property
    def is_calibrated(self) -> bool:
        """RM65 不需要标定"""
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        """RM65 不需要标定"""
        pass

    def configure(self) -> None:
        """配置拖动示教模式"""
        self.left_arm.configure()
        self.right_arm.configure()

    def setup_motors(self) -> None:
        """RM65 不需要电机设置"""
        pass

    def get_action(self) -> dict[str, float]:
        """
        读取左右臂当前关节角度
        
        Returns:
            dict: 左臂和右臂的关节角度,分别带 "left_" 和 "right_" 前缀
        """
        action_dict = {}

        # 获取左臂动作 (添加 "left_" 前缀)
        left_action = self.left_arm.get_action()
        action_dict.update({f"left_{key}": value for key, value in left_action.items()})

        # 获取右臂动作 (添加 "right_" 前缀)
        right_action = self.right_arm.get_action()
        action_dict.update({f"right_{key}": value for key, value in right_action.items()})

        return action_dict

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """RM65 不支持力反馈"""
        pass

    def disconnect(self) -> None:
        """断开左右臂连接并停止拖动示教"""
        self.left_arm.disconnect()
        self.right_arm.disconnect()