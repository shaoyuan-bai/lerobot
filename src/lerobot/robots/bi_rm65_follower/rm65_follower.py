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
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_bi_rm65_follower import RM65FollowerConfig
from .epg_gripper import EPGGripperClient

logger = logging.getLogger(__name__)


class RM65Follower(Robot):
    """
    睿尔曼 RM65 机械臂
    
    通过 TCP/IP 网络连接控制睿尔曼 RM65 六轴机械臂。
    使用睿尔曼官方 Python SDK (Robotic_Arm) 进行通信。
    """

    config_class = RM65FollowerConfig
    name = "rm65_follower"

    def __init__(self, config: RM65FollowerConfig):
        super().__init__(config)
        self.config = config

        # 延迟导入，避免在未安装 SDK 时报错
        try:
            from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

            # 初始化为三线程模式 (命令+接收+UDP监控)
            self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
            self._RoboticArm = RoboticArm
        except ImportError as e:
            raise ImportError(
                "睿尔曼 SDK 未安装。请运行: pip install Robotic_Arm\n"
                "或访问: https://github.com/RealManRobot/RM_API2"
            ) from e

        self.handle = None
        self.cameras = make_cameras_from_configs(config.cameras)

        # RM65 有 6 个关节
        self.joint_names = [f"joint_{i}" for i in range(1, 7)]
        
        # 夹爪位置缓存（用于降低读取频率）
        self._gripper_pos_cache: Optional[float] = None
        self._gripper_read_counter: int = 0
        self._gripper_read_interval: int = 3  # 每3帧读取一次夹爪
        
        # 初始化夹爪（如果启用）
        self.gripper: EPGGripperClient | None = None
        if config.enable_gripper:
            try:
                self.gripper = EPGGripperClient(
                    ip=config.ip_address,
                    port=config.port,
                    device_id=config.gripper_device_id,
                    force=config.gripper_force,
                    speed=config.gripper_speed,
                )
                logger.info("Gripper initialized for RM65")
            except Exception as e:
                logger.warning(f"Failed to initialize gripper: {e}. Continuing without gripper.")
                self.gripper = None

    @property
    def _motors_ft(self) -> dict[str, type]:
        features = {f"{joint}.pos": float for joint in self.joint_names}
        # 如果启用夹爪，添加夹爪位置特征
        if self.config.enable_gripper:
            features["gripper.pos"] = float
        return features

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        cameras_connected = all(cam.is_connected for cam in self.cameras.values())
        gripper_connected = True if self.gripper is None else self.gripper.is_connected
        return self.handle is not None and cameras_connected and gripper_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # 连接机械臂
        self.handle = self.arm.rm_create_robot_arm(self.config.ip_address, self.config.port)
        
        # 注意: 即使 handle.id == -1 (软件版本获取失败), SDK 的其他功能仍可能正常工作
        if self.handle.id == -1:
            logger.warning(
                f"RM65 at {self.config.ip_address}:{self.config.port} 连接时获取软件信息失败 (handle.id=-1), "
                f"但将继续尝试操作。如果后续操作失败,请检查: "
                f"1) 机械臂是否开机 2) 是否按下了使能按钮 3) Web界面(http://{self.config.ip_address}:80)是否需要设置"
            )
        else:
            logger.info(
                f"Connected to RM65 at {self.config.ip_address}:{self.config.port}, handle ID: {self.handle.id}"
            )

        # 连接相机
        for cam in self.cameras.values():
            cam.connect()
        
        # 连接夹爪（如果启用）
        if self.gripper is not None:
            try:
                self.gripper.connect()
                logger.info("Gripper connected successfully")
            except Exception as e:
                logger.warning(f"Failed to connect gripper: {e}. Continuing without gripper.")
                self.gripper = None

        self.configure()

    @property
    def is_calibrated(self) -> bool:
        # RM65 通过网络控制，不需要像串口电机那样的硬件校准
        return True

    def calibrate(self) -> None:
        # RM65 不需要物理校准过程
        pass

    def configure(self) -> None:
        """配置机械臂为真实模式"""
        ret = self.arm.rm_set_arm_run_mode(1)  # 1: 真实模式, 0: 仿真模式
        if ret != 0:
            logger.warning(f"Failed to set arm run mode to real mode: error code {ret}")

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()

        # 获取当前关节角度 (度)
        # 返回格式: (状态码, [joint1, joint2, joint3, joint4, joint5, joint6])
        ret, joint_angles = self.arm.rm_get_current_arm_state()

        if ret != 0:
            raise RuntimeError(f"Failed to get joint state from RM65: error code {ret}")

        # 提取关节角度列表 (假设返回的是包含关节数据的字典或列表)
        if isinstance(joint_angles, dict) and "joint" in joint_angles:
            angles_list = joint_angles["joint"]
        elif isinstance(joint_angles, (list, tuple)):
            angles_list = joint_angles
        else:
            # 如果返回格式不同，尝试直接访问
            angles_list = [joint_angles.get(f"joint_{i}", 0.0) for i in range(1, 7)]

        # 构建观察字典
        obs_dict = {f"{joint}.pos": float(angle) for joint, angle in zip(self.joint_names, angles_list)}
        
        # 读取夹爪位置（如果启用）
        if self.gripper is not None:
            # 按频率采样：每 N 帧读取一次，其他帧使用缓存
            self._gripper_read_counter += 1
            
            if self._gripper_read_counter >= self._gripper_read_interval or self._gripper_pos_cache is None:
                # 需要读取夹爪
                self._gripper_read_counter = 0
                
                # 使用清空缓冲区模式，确保数据正确
                gripper_pos_raw = self.gripper.get_position(skip_buffer_clear=False)
                
                if gripper_pos_raw is not None:
                    # 归一化: 0-255 -> 0-100
                    gripper_pos = (gripper_pos_raw / 255.0) * 100.0
                    self._gripper_pos_cache = float(gripper_pos)
                    obs_dict["gripper.pos"] = self._gripper_pos_cache
                    logger.debug(f"Gripper position read (fresh): {gripper_pos_raw} -> {gripper_pos:.1f}")
                else:
                    # 读取失败，使用缓存或默认值
                    if self._gripper_pos_cache is not None:
                        obs_dict["gripper.pos"] = self._gripper_pos_cache
                        logger.debug("Gripper read failed, using cached value")
                    else:
                        obs_dict["gripper.pos"] = 50.0
                        self._gripper_pos_cache = 50.0
                        logger.warning("Gripper position read failed, using fallback value 50.0")
            else:
                # 使用缓存值
                obs_dict["gripper.pos"] = self._gripper_pos_cache
                logger.debug(f"Gripper position (cached): {self._gripper_pos_cache:.1f}")
        else:
            logger.debug("Gripper is None, skipping gripper position read")

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # 读取相机
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # 提取目标关节角度 (度)
        goal_angles = [action[f"{joint}.pos"] for joint in self.joint_names]

        # 发送关节空间运动命令 (非阻塞)
        # rm_movej(joint: list[float], v: int, r: int, connect: int, block: int)
        # - joint: 6个关节角度列表
        # - v: 速度 1-100
        # - r: 轨迹过渡参数 (0-100, 0表示不过渡)
        # - connect: 轨迹连接 0=不连接, 1=连接
        # - block: 阻塞模式 0=非阻塞, 1=阻塞等待完成
        ret = self.arm.rm_movej(
            goal_angles, self.config.move_speed, 0, 0, 0  # joint, v, r, connect, block
        )

        if ret != 0:
            logger.warning(f"Failed to send action to RM65: error code {ret}")
        
        # 控制夹爪（如果启用）
        if self.gripper is not None and "gripper.pos" in action:
            gripper_pos_normalized = action["gripper.pos"]  # 0-100
            # 反归一化: 0-100 -> 0-255
            gripper_pos_raw = int((gripper_pos_normalized / 100.0) * 255.0)
            gripper_pos_raw = max(0, min(255, gripper_pos_raw))  # 限制范围
            
            success = self.gripper.set_position(gripper_pos_raw)
            if not success:
                logger.warning("Failed to set gripper position")

        return action

    def disconnect(self):
        if not self.is_connected:
            return

        # 断开机械臂连接
        if self.handle is not None:
            self.arm.rm_delete_robot_arm()
            self.handle = None
        
        # 断开夹爪（如果启用）
        if self.gripper is not None:
            self.gripper.disconnect()

        # 断开相机
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
