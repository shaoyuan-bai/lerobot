#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Utility functions for robot instantiation."""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lerobot.robots.config import RobotConfig
    from lerobot.robots.robot import Robot

logger = logging.getLogger(__name__)


def ensure_safe_goal_position(
    goal_present_pos: dict[str, tuple[float, float]],
    max_relative_target: float,
) -> dict[str, float]:
    """
    Ensures that goal positions are within a safe distance from present positions.
    
    Caps the magnitude of the relative action target by `max_relative_target` to avoid that
    the robot moves too fast which can damage the motors or the robot itself.
    
    Args:
        goal_present_pos: Dictionary mapping motor names to (goal_position, present_position) tuples.
        max_relative_target: Maximum allowed relative movement per motor.
        
    Returns:
        Dictionary mapping motor names to safe goal positions.
    """
    safe_goal_pos = {}
    for motor_name, (goal_pos, present_pos) in goal_present_pos.items():
        delta_pos = goal_pos - present_pos
        if abs(delta_pos) > max_relative_target:
            safe_delta_pos = max_relative_target if delta_pos > 0 else -max_relative_target
            safe_goal_pos[motor_name] = present_pos + safe_delta_pos
            logger.warning(
                f"Motor '{motor_name}': goal position {goal_pos:.2f} is too far from present "
                f"position {present_pos:.2f}. Capping to {safe_goal_pos[motor_name]:.2f} "
                f"(max_relative_target={max_relative_target})"
            )
        else:
            safe_goal_pos[motor_name] = goal_pos
    return safe_goal_pos


def make_robot_from_config(robot_config: "RobotConfig") -> "Robot":
    """
    Factory function to instantiate a Robot from a RobotConfig.
    
    Args:
        robot_config: Configuration object for the robot.
        
    Returns:
        An instantiated Robot object.
    """
    robot_type = robot_config.type
    
    if robot_type == "so100_follower":
        from lerobot.robots.so100_follower import SO100Follower
        return SO100Follower(robot_config)
    elif robot_type == "so101_follower":
        from lerobot.robots.so101_follower import SO101Follower
        return SO101Follower(robot_config)
    elif robot_type == "bi_so100_follower":
        from lerobot.robots.bi_so100_follower import BiSO100Follower
        return BiSO100Follower(robot_config)
    elif robot_type == "bi_so101_follower":
        from lerobot.robots.bi_so101_follower import BiSO101Follower
        return BiSO101Follower(robot_config)
    elif robot_type == "bi_rm65_follower":
        from lerobot.robots.bi_rm65_follower import BiRM65Follower
        return BiRM65Follower(robot_config)
    elif robot_type == "rm65_follower":
        from lerobot.robots.bi_rm65_follower import RM65Follower
        return RM65Follower(robot_config)
    elif robot_type == "koch_follower":
        from lerobot.robots.koch_follower import KochFollower
        return KochFollower(robot_config)
    elif robot_type == "lekiwi":
        from lerobot.robots.lekiwi import Lekiwi
        return Lekiwi(robot_config)
    elif robot_type == "hope_jr":
        from lerobot.robots.hope_jr import HopeJR
        return HopeJR(robot_config)
    elif robot_type == "reachy2":
        from lerobot.robots.reachy2 import Reachy2
        return Reachy2(robot_config)
    elif robot_type == "stretch3":
        from lerobot.robots.stretch3 import Stretch3
        return Stretch3(robot_config)
    elif robot_type == "viperx":
        from lerobot.robots.viperx import ViperX
        return ViperX(robot_config)
    elif robot_type == "omni_base":
        from lerobot.robots.omni_base import OmniBase
        return OmniBase(robot_config)
    else:
        raise ValueError(
            f"Unknown robot type '{robot_type}'. Available types: "
            "so100_follower, so101_follower, bi_so100_follower, bi_so101_follower, "
            "rm65_follower, bi_rm65_follower, koch_follower, lekiwi, hope_jr, reachy2, stretch3, "
            "viperx, omni_base"
        )
