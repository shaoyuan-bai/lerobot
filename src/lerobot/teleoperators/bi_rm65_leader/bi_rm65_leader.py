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
        self.robot_type = config.type

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
        return True  # 复用 follower 连接

    def connect(self) -> None:
        """不需要连接（复用 follower）"""
        pass

    @property
    def is_calibrated(self) -> bool:
        return True

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
        """拖动示教：返回 follower observation 作为 action"""
        # record.py 会在调用后使用 obs，所以这里返回空字典
        # 实际 action 由 record.py 的 teleop_action_processor 从 obs 提取
        return {}

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """RM65 不支持力反馈"""
        pass

    def disconnect(self) -> None:
        """不需要断开（复用 follower）"""
        pass