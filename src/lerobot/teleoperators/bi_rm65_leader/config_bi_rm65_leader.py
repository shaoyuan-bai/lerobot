from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("rm65_leader")
@dataclass
class RM65LeaderConfig(TeleoperatorConfig):
    """睿尔曼 RM65 单臂主臂配置 (拖动示教)"""
    
    arm_ip: str
    port: int = 8080
    drag_sensitivity: int = 5  # 拖动灵敏度 1-10


@TeleoperatorConfig.register_subclass("bi_rm65_leader")
@dataclass
class BiRM65LeaderConfig(TeleoperatorConfig):
    """睿尔曼 RM65 双臂主臂配置 (用于手动拖动录制)"""
    
    left_arm_ip: str
    right_arm_ip: str
    port: int = 8080
    drag_sensitivity: int = 5  # 拖动灵敏度 1-10
