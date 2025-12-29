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
from typing import Any

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_bi_rm65_leader import RM65LeaderConfig

logger = logging.getLogger(__name__)


class RM65Leader(Teleoperator):
    """
    睿尔曼 RM65 单臂主臂 (通过拖动示教读取关节角度)
    
    使用 RM65 的拖动示教功能,允许手动移动机械臂并读取关节角度。
    """

    config_class = RM65LeaderConfig
    name = "rm65_leader"

    def __init__(self, config: RM65LeaderConfig):
        super().__init__(config)
        self.config = config
        self.handle = None
        self.arm = None
        
        # 关节名称 (RM65 有6个关节)
        self.joint_names = [f"joint_{i}" for i in range(1, 7)]

    @property
    def action_features(self) -> dict[str, type]:
        """返回动作特征 (6个关节角度)"""
        return {f"{joint}.pos": float for joint in self.joint_names}

    @property
    def feedback_features(self) -> dict[str, type]:
        """RM65 不支持力反馈"""
        return {}

    @property
    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self.handle is not None and self.arm is not None

    def connect(self, calibrate: bool = True) -> None:
        """
        连接到 RM65 主臂并启动拖动示教模式
        
        Args:
            calibrate: RM65 不需要标定,忽略此参数
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        try:
            from Robotic_Arm.rm_robot_interface import RoboticArm
        except ImportError as e:
            raise ImportError(
                "睿尔曼 SDK 未安装。请运行: pip install Robotic_Arm"
            ) from e

        # 创建机械臂实例
        self.arm = RoboticArm()
        
        # 连接到机械臂
        self.handle = self.arm.rm_create_robot_arm(self.config.arm_ip, self.config.port)
        
        if self.handle != 0:
            raise ConnectionError(
                f"Failed to connect to RM65 at {self.config.arm_ip}:{self.config.port}. "
                f"Error code: {self.handle}"
            )
        
        # 配置拖动示教
        self.configure()
        
        logger.info(f"{self} connected at {self.config.arm_ip}:{self.config.port}")
        logger.info(f"拖动示教已启动,灵敏度: {self.config.drag_sensitivity}")

    @property
    def is_calibrated(self) -> bool:
        """RM65 使用绝对编码器,不需要标定"""
        return True

    def calibrate(self) -> None:
        """RM65 不需要标定"""
        pass

    def configure(self) -> None:
        """配置拖动示教模式"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        
        # 设置拖动灵敏度 (1-10, 数值越大越灵敏)
        ret = self.arm.rm_set_drag_teach_sensitivity(self.config.drag_sensitivity)
        if ret != 0:
            logger.warning(f"Failed to set drag sensitivity: error code {ret}")
        
        # 启动拖动示教模式
        ret = self.arm.rm_start_drag_teach()
        if ret != 0:
            raise RuntimeError(f"Failed to start drag teach mode: error code {ret}")
        
        logger.info("拖动示教模式已启动,可以手动移动机械臂")

    def setup_motors(self) -> None:
        """RM65 不需要电机设置"""
        pass

    def get_action(self) -> dict[str, float]:
        """
        读取当前关节角度 (拖动示教模式下)
        
        Returns:
            dict: 格式为 {"joint_1.pos": angle1, ..., "joint_6.pos": angle6}
        """
        start = time.perf_counter()
        
        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self} is not connected. Run connect() first."
            )
        
        # 读取当前关节角度
        joint_status = self.arm.rm_get_joint_degree()
        
        if not isinstance(joint_status, dict) or "joint" not in joint_status:
            logger.warning(f"Invalid joint status response: {joint_status}")
            # 返回零角度作为fallback
            return {f"{joint}.pos": 0.0 for joint in self.joint_names}
        
        # 提取关节角度 (度)
        joint_angles = joint_status["joint"]
        
        if len(joint_angles) != 6:
            logger.warning(f"Expected 6 joints, got {len(joint_angles)}")
        
        # 构建动作字典
        action = {
            f"{joint}.pos": float(joint_angles[i])
            for i, joint in enumerate(self.joint_names)
        }
        
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """RM65 不支持力反馈"""
        pass

    def disconnect(self) -> None:
        """断开连接并停止拖动示教"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        
        # 停止拖动示教
        ret = self.arm.rm_stop_drag_teach()
        if ret != 0:
            logger.warning(f"Failed to stop drag teach: error code {ret}")
        
        # 断开连接
        if self.handle is not None:
            self.arm.rm_delete_robot_arm()
            self.handle = None
            self.arm = None
        
        logger.info(f"{self} disconnected")
