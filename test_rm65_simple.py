#!/usr/bin/env python3
"""RM65 简化测试 - 直接读取关节角度"""

from Robotic_Arm.rm_robot_interface import RoboticArm
import time

def test_simple_read():
    """简单测试: 连接并读取关节角度"""
    ip = "169.254.128.20"
    port = 8080
    
    print("=" * 60)
    print("RM65 简化测试")
    print("=" * 60)
    print()
    
    # 创建机械臂实例
    print(f"1️⃣  连接到 {ip}:{port}...")
    arm = RoboticArm()
    handle = arm.rm_create_robot_arm(ip, port)
    print(f"   handle.id: {handle.id}")
    print()
    
    # 尝试读取关节角度
    print("2️⃣  尝试读取关节角度...")
    for i in range(5):
        try:
            # 方法1: rm_get_joint_degree
            print(f"\n   尝试 {i+1}: rm_get_joint_degree()")
            result = arm.rm_get_joint_degree()
            print(f"   返回值: {result}")
            
            if isinstance(result, dict) and "joint" in result:
                print(f"   ✓ 成功! 关节角度: {result['joint']}")
                break
            elif isinstance(result, tuple) and len(result) >= 2:
                ret_code, data = result[0], result[1]
                print(f"   返回码: {ret_code}, 数据: {data}")
                if ret_code == 0:
                    print(f"   ✓ 成功!")
                    break
            else:
                print(f"   格式不符,等待1秒后重试...")
            
            time.sleep(1)
            
        except Exception as e:
            print(f"   ✗ 异常: {e}")
            time.sleep(1)
    
    print()
    print("3️⃣  尝试其他读取方法...")
    
    # 方法2: rm_get_current_arm_state
    try:
        print("\n   尝试: rm_get_current_arm_state()")
        result = arm.rm_get_current_arm_state()
        print(f"   返回值: {result}")
    except Exception as e:
        print(f"   ✗ 异常: {e}")
    
    # 方法3: rm_get_joint_state
    try:
        print("\n   尝试: rm_get_joint_state()")
        result = arm.rm_get_joint_state()
        print(f"   返回值: {result}")
    except Exception as e:
        print(f"   ✗ 异常: {e}")
    
    print()
    print("4️⃣  断开连接...")
    arm.rm_delete_robot_arm()
    print("   ✓ 已断开")
    
    print()
    print("=" * 60)
    print("提示:")
    print("  - 如果所有方法都返回错误码 -2,说明机械臂未使能")
    print("  - 请按下机械臂的使能按钮后重试")
    print("  - 或者访问 Web 界面 http://169.254.128.20:80 进行设置")
    print("=" * 60)

if __name__ == "__main__":
    test_simple_read()
