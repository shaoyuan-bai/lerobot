#!/usr/bin/env python
"""
测试 RM65 拖动示教功能

使用方法:
1. 确保两台 RM65 已上电并联网
2. 运行: python test_rm65_drag_teach.py
3. 手动拖动左臂,观察控制台输出的关节角度
"""

import time
from lerobot.teleoperators.bi_rm65_leader import BiRM65Leader, BiRM65LeaderConfig


def test_drag_teach():
    print("=" * 60)
    print("RM65 拖动示教测试")
    print("=" * 60)
    
    # 创建配置
    config = BiRM65LeaderConfig(
        id="test_rm65_leader",
        left_arm_ip="169.254.128.20",
        right_arm_ip="169.254.128.21",
        port=8080,
        drag_sensitivity=5,  # 灵敏度 1-10
    )
    
    print(f"\n配置:")
    print(f"  左臂: {config.left_arm_ip}:{config.port}")
    print(f"  右臂: {config.right_arm_ip}:{config.port}")
    print(f"  拖动灵敏度: {config.drag_sensitivity}")
    
    # 创建主臂实例
    leader = BiRM65Leader(config)
    
    try:
        print("\n正在连接并启动拖动示教...")
        leader.connect(calibrate=False)
        
        print("\n✓ 拖动示教已启动!")
        print("\n" + "=" * 60)
        print("🖐️  现在可以手动拖动机械臂了!")
        print("=" * 60)
        print("\n按 Ctrl+C 停止\n")
        
        # 循环读取并显示关节角度
        frame_count = 0
        while True:
            # 读取动作 (关节角度)
            action = leader.get_action()
            
            frame_count += 1
            if frame_count % 10 == 0:  # 每10帧显示一次
                print(f"\r帧 {frame_count}:", end=" ")
                
                # 显示左臂
                print("左臂[", end="")
                for i in range(1, 7):
                    angle = action.get(f"left_joint_{i}.pos", 0.0)
                    print(f"{angle:6.1f}°", end=" ")
                print("] ", end="")
                
                # 显示右臂
                print("右臂[", end="")
                for i in range(1, 7):
                    angle = action.get(f"right_joint_{i}.pos", 0.0)
                    print(f"{angle:6.1f}°", end=" ")
                print("]", end="", flush=True)
            
            time.sleep(0.1)  # 10Hz
    
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if leader.is_connected:
            print("\n\n正在停止拖动示教并断开连接...")
            leader.disconnect()
            print("✓ 已断开")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    test_drag_teach()
