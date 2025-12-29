#!/usr/bin/env python
"""
RM65 双臂交互测试脚本 (适合 SSH 远程使用)

功能:
1. 读取当前关节角度
2. 控制单个关节移动
3. 循环录制关节数据
4. 回放录制的轨迹

使用方法:
    python test_rm65_interactive.py
"""

import time
import json
from pathlib import Path
from lerobot.robots.bi_rm65_follower import BiRM65FollowerConfig, BiRM65Follower


def print_joint_states(robot):
    """打印当前关节状态"""
    obs = robot.get_observation()
    
    print("\n" + "=" * 60)
    print("当前关节角度:")
    print("=" * 60)
    
    print("\n【左臂】")
    for joint in robot.left_arm.joint_names:
        key = f"left_{joint}.pos"
        print(f"  {joint}: {obs[key]:7.2f}°")
    
    print("\n【右臂】")
    for joint in robot.right_arm.joint_names:
        key = f"right_{joint}.pos"
        print(f"  {joint}: {obs[key]:7.2f}°")
    print("=" * 60)


def move_single_joint(robot):
    """移动单个关节"""
    print("\n" + "=" * 60)
    print("单关节移动测试")
    print("=" * 60)
    
    obs = robot.get_observation()
    
    print("\n可用关节:")
    joints = list(robot.action_features.keys())
    for i, joint in enumerate(joints, 1):
        current_angle = obs.get(joint, 0.0)
        print(f"  {i:2d}. {joint:25s} (当前: {current_angle:7.2f}°)")
    
    try:
        choice = input("\n选择要移动的关节 (输入编号或名称,回车跳过): ").strip()
        if not choice:
            return
        
        # 解析选择
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(joints):
                joint_name = joints[idx]
            else:
                print("✗ 无效的编号")
                return
        else:
            joint_name = choice if choice in joints else None
            if not joint_name:
                print("✗ 无效的关节名称")
                return
        
        # 获取目标角度
        current = obs[joint_name]
        print(f"\n当前角度: {current:.2f}°")
        
        delta = input("输入角度变化量 (如 +10 或 -5): ").strip()
        if not delta:
            return
        
        target = current + float(delta)
        print(f"目标角度: {target:.2f}°")
        
        # 确认
        confirm = input("确认执行? (y/n): ").strip().lower()
        if confirm != 'y':
            print("已取消")
            return
        
        # 构建动作
        action = obs.copy()
        action[joint_name] = target
        
        # 发送动作
        print("\n正在移动...")
        robot.send_action(action)
        time.sleep(2)
        
        # 读取新状态
        new_obs = robot.get_observation()
        actual = new_obs[joint_name]
        print(f"✓ 移动完成! 实际角度: {actual:.2f}°")
        
    except ValueError:
        print("✗ 输入格式错误")
    except KeyboardInterrupt:
        print("\n已取消")


def record_trajectory(robot):
    """录制轨迹"""
    print("\n" + "=" * 60)
    print("轨迹录制")
    print("=" * 60)
    
    try:
        duration = input("\n录制时长 (秒,默认10): ").strip()
        duration = int(duration) if duration else 10
        
        fps = input("采样频率 (Hz,默认10): ").strip()
        fps = int(fps) if fps else 10
        
        print(f"\n将录制 {duration} 秒,每秒 {fps} 帧")
        print("请手动移动机械臂到期望位置...")
        input("按回车开始录制...")
        
        trajectory = []
        interval = 1.0 / fps
        samples = duration * fps
        
        print(f"\n🔴 录制中... (共 {samples} 帧)")
        
        for i in range(samples):
            obs = robot.get_observation()
            timestamp = time.time()
            
            # 保存关节角度
            frame = {
                'timestamp': timestamp,
                'joints': {k: v for k, v in obs.items() if k.endswith('.pos')}
            }
            trajectory.append(frame)
            
            # 进度显示
            if (i + 1) % fps == 0:
                print(f"  已录制 {i + 1}/{samples} 帧 ({(i+1)/fps:.1f}s)")
            
            time.sleep(interval)
        
        print(f"\n✓ 录制完成! 共 {len(trajectory)} 帧")
        
        # 保存到文件
        save = input("\n是否保存轨迹? (y/n): ").strip().lower()
        if save == 'y':
            filename = input("文件名 (默认 trajectory.json): ").strip()
            filename = filename if filename else "trajectory.json"
            
            with open(filename, 'w') as f:
                json.dump(trajectory, f, indent=2)
            
            print(f"✓ 已保存到 {filename}")
            return filename
        
        return None
        
    except KeyboardInterrupt:
        print("\n\n✗ 录制已中断")
        return None
    except ValueError:
        print("✗ 输入格式错误")
        return None


def replay_trajectory(robot):
    """回放轨迹"""
    print("\n" + "=" * 60)
    print("轨迹回放")
    print("=" * 60)
    
    # 列出可用文件
    json_files = list(Path('.').glob('*.json'))
    if not json_files:
        print("\n✗ 未找到轨迹文件")
        return
    
    print("\n可用轨迹文件:")
    for i, f in enumerate(json_files, 1):
        size = f.stat().st_size / 1024
        print(f"  {i}. {f.name} ({size:.1f} KB)")
    
    try:
        choice = input("\n选择文件 (输入编号或文件名): ").strip()
        
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(json_files):
                filename = json_files[idx]
            else:
                print("✗ 无效的编号")
                return
        else:
            filename = Path(choice)
            if not filename.exists():
                print("✗ 文件不存在")
                return
        
        # 加载轨迹
        with open(filename, 'r') as f:
            trajectory = json.load(f)
        
        print(f"\n✓ 已加载 {len(trajectory)} 帧")
        
        # 确认回放
        confirm = input("开始回放? (y/n): ").strip().lower()
        if confirm != 'y':
            print("已取消")
            return
        
        print("\n▶ 回放中...")
        
        for i, frame in enumerate(trajectory):
            # 发送动作
            robot.send_action(frame['joints'])
            
            # 进度显示
            if (i + 1) % 10 == 0:
                progress = (i + 1) / len(trajectory) * 100
                print(f"  进度: {i+1}/{len(trajectory)} ({progress:.1f}%)")
            
            # 等待下一帧
            if i < len(trajectory) - 1:
                dt = trajectory[i + 1]['timestamp'] - frame['timestamp']
                time.sleep(max(0.01, dt))
        
        print("\n✓ 回放完成!")
        
    except KeyboardInterrupt:
        print("\n\n✗ 回放已中断")
    except Exception as e:
        print(f"\n✗ 错误: {e}")


def main_menu(robot):
    """主菜单"""
    while True:
        print("\n" + "=" * 60)
        print("RM65 双臂交互测试")
        print("=" * 60)
        print("\n1. 显示关节状态")
        print("2. 移动单个关节")
        print("3. 录制轨迹")
        print("4. 回放轨迹")
        print("0. 退出")
        print("=" * 60)
        
        choice = input("\n请选择操作: ").strip()
        
        if choice == '1':
            print_joint_states(robot)
        elif choice == '2':
            move_single_joint(robot)
        elif choice == '3':
            record_trajectory(robot)
        elif choice == '4':
            replay_trajectory(robot)
        elif choice == '0':
            print("\n再见!")
            break
        else:
            print("\n✗ 无效的选择")


def main():
    """主函数"""
    print("=" * 60)
    print("RM65 双臂交互测试 (SSH 远程模式)")
    print("=" * 60)
    
    # 配置机器人
    config = BiRM65FollowerConfig(
        id="rm65_interactive",
        left_arm_ip="169.254.128.20",
        right_arm_ip="169.254.128.21",
        port=8080,
        move_speed=30,
        cameras={},
    )
    
    print(f"\n配置:")
    print(f"  左臂: {config.left_arm_ip}:{config.port}")
    print(f"  右臂: {config.right_arm_ip}:{config.port}")
    print(f"  速度: {config.move_speed}")
    
    # 创建并连接
    robot = BiRM65Follower(config)
    
    try:
        print("\n正在连接...")
        robot.connect(calibrate=False)
        print("✓ 连接成功!")
        
        # 显示初始状态
        print_joint_states(robot)
        
        # 进入主菜单
        main_menu(robot)
        
    except KeyboardInterrupt:
        print("\n\n程序被中断")
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if robot.is_connected:
            print("\n正在断开连接...")
            robot.disconnect()
            print("✓ 已断开")


if __name__ == "__main__":
    main()
