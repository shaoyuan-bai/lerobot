#!/usr/bin/env python
"""
测试睿尔曼 RM65 双臂连接的脚本

使用方法:
1. 确保已安装睿尔曼 SDK: pip install Robotic_Arm
2. 修改下方的 IP 地址为你的机械臂实际 IP
3. 运行: python test_rm65_connection.py
"""

import time
from lerobot.robots.bi_rm65_follower import BiRM65FollowerConfig, BiRM65Follower

def test_connection():
    """测试双臂连接"""
    
    print("=" * 60)
    print("睿尔曼 RM65 双臂连接测试")
    print("=" * 60)
    
    # 配置机器人 (请修改为你的实际 IP 地址)
    config = BiRM65FollowerConfig(
        id="test_rm65",
        left_arm_ip="192.168.1.10",   # 左臂 IP - 请修改
        right_arm_ip="192.168.1.11",  # 右臂 IP - 请修改
        port=8080,
        move_speed=30,  # 较慢的速度用于测试
        cameras={},  # 暂不测试相机
    )
    
    print(f"\n配置信息:")
    print(f"  左臂 IP: {config.left_arm_ip}:{config.port}")
    print(f"  右臂 IP: {config.right_arm_ip}:{config.port}")
    print(f"  运动速度: {config.move_speed}")
    
    # 创建机器人实例
    print("\n正在创建机器人实例...")
    robot = BiRM65Follower(config)
    
    # 检查特征定义
    print("\n✓ 机器人实例创建成功")
    print(f"\n观察特征 (observation_features):")
    for key in list(robot.observation_features.keys())[:6]:
        print(f"  - {key}: {robot.observation_features[key]}")
    print(f"  ... (共 {len(robot.observation_features)} 个特征)")
    
    print(f"\n动作特征 (action_features):")
    for key in list(robot.action_features.keys())[:6]:
        print(f"  - {key}: {robot.action_features[key]}")
    print(f"  ... (共 {len(robot.action_features)} 个特征)")
    
    # 尝试连接
    print("\n" + "-" * 60)
    print("开始连接机械臂...")
    print("-" * 60)
    
    try:
        robot.connect(calibrate=False)
        print("✓ 连接成功!")
        
        # 检查连接状态
        if robot.is_connected:
            print("✓ 连接状态确认: 已连接")
        else:
            print("✗ 连接状态异常")
            return
        
        # 读取当前状态
        print("\n正在读取机械臂状态...")
        obs = robot.get_observation()
        
        print(f"\n当前左臂关节角度:")
        for joint in robot.left_arm.joint_names:
            key = f"left_{joint}.pos"
            print(f"  {key}: {obs[key]:.2f}°")
        
        print(f"\n当前右臂关节角度:")
        for joint in robot.right_arm.joint_names:
            key = f"right_{joint}.pos"
            print(f"  {key}: {obs[key]:.2f}°")
        
        print("\n✓ 状态读取成功!")
        
        # 测试发送动作 (小幅度移动)
        print("\n" + "-" * 60)
        print("测试发送动作命令...")
        print("将发送当前位置作为目标 (不移动，仅测试通信)")
        print("-" * 60)
        
        # 使用当前位置作为目标
        test_action = {key: value for key, value in obs.items() if key.endswith('.pos')}
        
        sent_action = robot.send_action(test_action)
        print("✓ 动作发送成功!")
        
        time.sleep(1)
        
    except ImportError as e:
        print(f"\n✗ 导入错误: {e}")
        print("\n请安装睿尔曼 SDK:")
        print("  pip install Robotic_Arm")
        return
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return
        
    finally:
        # 断开连接
        if robot.is_connected:
            print("\n正在断开连接...")
            robot.disconnect()
            print("✓ 已断开连接")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    test_connection()
