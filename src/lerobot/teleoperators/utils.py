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
"""Utility functions and constants for teleoperators."""

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lerobot.teleoperators.config import TeleoperatorConfig
    from lerobot.teleoperators.teleoperator import Teleoperator


class TeleopEvents(str, Enum):
    """Events that can be triggered during teleoperation."""
    
    IS_INTERVENTION = "is_intervention"
    TERMINATE_EPISODE = "terminate_episode"
    SUCCESS = "success"
    FAILURE = "failure"
    RERECORD_EPISODE = "rerecord_episode"


def make_teleoperator_from_config(teleop_config: "TeleoperatorConfig") -> "Teleoperator":
    """
    Factory function to instantiate a Teleoperator from a TeleoperatorConfig.
    
    Args:
        teleop_config: Configuration object for the teleoperator.
        
    Returns:
        An instantiated Teleoperator object.
    """
    teleop_type = teleop_config.type
    
    if teleop_type == "gamepad":
        from lerobot.teleoperators.gamepad import GamepadTeleop
        return GamepadTeleop(teleop_config)
    elif teleop_type == "keyboard":
        from lerobot.teleoperators.keyboard import KeyboardTeleop
        return KeyboardTeleop(teleop_config)
    elif teleop_type == "phone":
        from lerobot.teleoperators.phone import Phone
        return Phone(teleop_config)
    elif teleop_type == "koch_leader":
        from lerobot.teleoperators.koch_leader import KochLeader
        return KochLeader(teleop_config)
    elif teleop_type == "so100_leader":
        from lerobot.teleoperators.so100_leader import SO100Leader
        return SO100Leader(teleop_config)
    elif teleop_type == "so101_leader":
        from lerobot.teleoperators.so101_leader import SO101Leader
        return SO101Leader(teleop_config)
    elif teleop_type == "bi_so100_leader":
        from lerobot.teleoperators.bi_so100_leader import BiSO100Leader
        return BiSO100Leader(teleop_config)
    elif teleop_type == "bi_so101_leader":
        from lerobot.teleoperators.bi_so101_leader import BiSO101Leader
        return BiSO101Leader(teleop_config)
    elif teleop_type == "bi_rm65_leader":
        from lerobot.teleoperators.bi_rm65_leader import BiRM65Leader
        return BiRM65Leader(teleop_config)
    elif teleop_type == "rm65_leader":
        from lerobot.teleoperators.bi_rm65_leader import RM65Leader
        return RM65Leader(teleop_config)
    elif teleop_type == "homunculus":
        from lerobot.teleoperators.homunculus import Homunculus
        return Homunculus(teleop_config)
    elif teleop_type == "reachy2_teleoperator":
        from lerobot.teleoperators.reachy2_teleoperator import Reachy2Teleoperator
        return Reachy2Teleoperator(teleop_config)
    elif teleop_type == "stretch3_gamepad":
        from lerobot.teleoperators.stretch3_gamepad import Stretch3Gamepad
        return Stretch3Gamepad(teleop_config)
    elif teleop_type == "widowx":
        from lerobot.teleoperators.widowx import WidowX
        return WidowX(teleop_config)
    elif teleop_type == "xlebi_so101_leader":
        from lerobot.teleoperators.xlebi_so101_leader import XlebiSO101Leader
        return XlebiSO101Leader(teleop_config)
    else:
        raise ValueError(
            f"Unknown teleoperator type '{teleop_type}'. Available types: "
            "gamepad, keyboard, phone, koch_leader, so100_leader, so101_leader, "
            "bi_so100_leader, bi_so101_leader, rm65_leader, bi_rm65_leader, homunculus, "
            "reachy2_teleoperator, stretch3_gamepad, widowx, xlebi_so101_leader"
        )
