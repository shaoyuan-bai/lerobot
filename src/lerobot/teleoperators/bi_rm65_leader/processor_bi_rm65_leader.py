"""RM65 拖动示教：从 observation 提取 action"""


def rm65_drag_teach_processor(data: tuple) -> dict[str, float]:
    """
    拖动示教处理器：直接使用 observation 作为 action
    
    Args:
        data: (teleop_action, robot_observation)
        
    Returns:
        机器人 observation 中的位置数据（作为 action）
    """
    _, obs = data
    
    # 提取所有 .pos 键作为 action
    action = {key: value for key, value in obs.items() if ".pos" in key and "image" not in key}
    
    return action
