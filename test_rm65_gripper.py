#!/usr/bin/env python

"""
测试 RM65 右臂夹爪功能

测试内容:
1. 连接夹爪
2. 读取当前位置
3. 控制夹爪开合
4. 验证数据采集
"""

import time
from src.lerobot.robots.bi_rm65_follower.epg_gripper import EPGGripperClient

def test_gripper_standalone():
    """测试独立夹爪控制"""
    print("=" * 60)
    print("测试 EPG 夹爪独立控制")
    print("=" * 60)
    
    # 初始化夹爪
    gripper = EPGGripperClient(
        ip="169.254.128.21",  # 右臂IP
        port=8080,
        device_id=9,
        force=60,
        speed=255,
    )
    
    # 连接
    print("\n1. 连接夹爪...")
    try:
        gripper.connect()
        print("✓ 夹爪连接成功")
    except Exception as e:
        print(f"✗ 夹爪连接失败: {e}")
        return
    
    # 读取位置
    print("\n2. 读取当前位置...")
    pos = gripper.get_position()
    if pos is not None:
        print(f"✓ 当前位置: {pos} (原始值: 0-255)")
        print(f"  归一化值: {pos / 255.0 * 100.0:.1f} (0-100)")
    else:
        print("✗ 读取位置失败")
    
    # 测试控制
    print("\n3. 测试夹爪控制...")
    test_positions = [
        (0, "完全张开"),
        (127, "中间位置"),
        (255, "完全闭合"),
        (127, "返回中间"),
    ]
    
    for pos, desc in test_positions:
        print(f"  → 移动到 {pos} ({desc})...")
        success = gripper.set_position(pos)
        if success:
            print(f"    ✓ 命令发送成功")
        else:
            print(f"    ✗ 命令发送失败")
        time.sleep(2)  # 等待运动完成
        
        # 读取实际位置
        actual_pos = gripper.get_position()
        if actual_pos is not None:
            print(f"    实际位置: {actual_pos}")
    
    # 断开连接
    print("\n4. 断开夹爪...")
    gripper.disconnect()
    print("✓ 测试完成")


def test_gripper_with_robot():
    """测试集成到机器人的夹爪控制"""
    print("\n" + "=" * 60)
    print("测试 RM65 机器人夹爪集成")
    print("=" * 60)
    
    from src.lerobot.robots.bi_rm65_follower import BiRM65Follower
    from src.lerobot.robots.bi_rm65_follower.config_bi_rm65_follower import BiRM65FollowerConfig
    
    # 创建配置
    config = BiRM65FollowerConfig(
        left_arm_ip="169.254.128.20",
        right_arm_ip="169.254.128.21",
        port=8080,
        move_speed=50,
        enable_right_gripper=True,  # 启用右臂夹爪
        gripper_device_id=9,
        gripper_force=60,
        gripper_speed=255,
        cameras={},
    )
    
    # 创建机器人
    robot = BiRM65Follower(config)
    
    # 检查特征
    print("\n1. 检查特征定义...")
    print(f"  Observation features: {list(robot.observation_features.keys())}")
    print(f"  Action features: {list(robot.action_features.keys())}")
    
    if "right_gripper.pos" in robot.observation_features:
        print("  ✓ right_gripper.pos 在 observation_features 中")
    else:
        print("  ✗ right_gripper.pos 不在 observation_features 中")
    
    if "right_gripper.pos" in robot.action_features:
        print("  ✓ right_gripper.pos 在 action_features 中")
    else:
        print("  ✗ right_gripper.pos 不在 action_features 中")
    
    # 连接机器人
    print("\n2. 连接机器人...")
    try:
        robot.connect(calibrate=False)
        print("  ✓ 机器人连接成功")
    except Exception as e:
        print(f"  ✗ 机器人连接失败: {e}")
        return
    
    # 获取观察
    print("\n3. 获取观察数据...")
    try:
        obs = robot.get_observation()
        if "right_gripper.pos" in obs:
            print(f"  ✓ 夹爪位置: {obs['right_gripper.pos']:.2f} (0-100)")
        else:
            print("  ✗ 观察数据中没有夹爪位置")
        
        # 显示所有关节位置
        print("\n  关节位置:")
        for key, value in obs.items():
            if key.endswith(".pos"):
                print(f"    {key}: {value:.2f}")
    except Exception as e:
        print(f"  ✗ 获取观察失败: {e}")
    
    # 测试动作
    print("\n4. 测试发送动作...")
    test_actions = [
        {"right_gripper.pos": 0.0},    # 完全张开
        {"right_gripper.pos": 50.0},   # 中间位置
        {"right_gripper.pos": 100.0},  # 完全闭合
        {"right_gripper.pos": 50.0},   # 返回中间
    ]
    
    for action in test_actions:
        print(f"  → 发送动作: {action}")
        try:
            # 构建完整动作（包含所有关节）
            full_action = obs.copy()  # 使用当前观察作为基础
            full_action.update(action)  # 更新夹爪位置
            
            robot.send_action(full_action)
            print(f"    ✓ 动作发送成功")
            time.sleep(2)
        except Exception as e:
            print(f"    ✗ 动作发送失败: {e}")
    
    # 断开连接
    print("\n5. 断开机器人...")
    robot.disconnect()
    print("  ✓ 测试完成")


if __name__ == "__main__":
    import sys
    
    print("RM65 夹爪功能测试")
    print("=" * 60)
    print("选择测试模式:")
    print("  1. 独立夹爪测试 (只测试夹爪通信)")
    print("  2. 机器人集成测试 (测试完整系统)")
    print("  3. 全部测试")
    
    choice = input("\n请选择 (1/2/3): ").strip()
    
    if choice == "1":
        test_gripper_standalone()
    elif choice == "2":
        test_gripper_with_robot()
    elif choice == "3":
        test_gripper_standalone()
        test_gripper_with_robot()
    else:
        print("无效选择，运行全部测试")
        test_gripper_standalone()
        test_gripper_with_robot()
    
    print("\n" + "=" * 60)
    print("测试结束")
    print("=" * 60)
