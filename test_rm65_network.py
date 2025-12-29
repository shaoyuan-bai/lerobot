#!/usr/bin/env python3
"""RM65 网络连接诊断工具"""

import subprocess
import socket
from Robotic_Arm.rm_robot_interface import RoboticArm

def test_network():
    """测试网络连接"""
    print("=" * 60)
    print("RM65 网络诊断工具")
    print("=" * 60)
    print()
    
    left_ip = "169.254.128.20"
    right_ip = "169.254.128.21"
    port = 8080
    
    # 1. 测试 ping 连通性
    print("1️⃣  测试 PING 连通性...")
    print("-" * 60)
    for name, ip in [("左臂", left_ip), ("右臂", right_ip)]:
        try:
            result = subprocess.run(
                ["ping", "-c", "1", "-W", "2", ip],
                capture_output=True,
                text=True,
                timeout=3
            )
            if result.returncode == 0:
                print(f"✓ {name} ({ip}): PING 成功")
            else:
                print(f"✗ {name} ({ip}): PING 失败")
                print(f"  输出: {result.stdout}")
        except Exception as e:
            print(f"✗ {name} ({ip}): PING 测试异常 - {e}")
    print()
    
    # 2. 测试 TCP 端口连接
    print("2️⃣  测试 TCP 端口 8080 连接...")
    print("-" * 60)
    for name, ip in [("左臂", left_ip), ("右臂", right_ip)]:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        try:
            result = sock.connect_ex((ip, port))
            if result == 0:
                print(f"✓ {name} ({ip}:{port}): TCP 端口可连接")
            else:
                print(f"✗ {name} ({ip}:{port}): TCP 端口不可连接 (错误码: {result})")
        except Exception as e:
            print(f"✗ {name} ({ip}:{port}): TCP 连接异常 - {e}")
        finally:
            sock.close()
    print()
    
    # 3. 测试 SDK 连接(详细模式)
    print("3️⃣  测试 SDK 连接...")
    print("-" * 60)
    
    arm = RoboticArm()
    
    for name, ip in [("左臂", left_ip), ("右臂", right_ip)]:
        print(f"\n测试 {name} ({ip}:{port}):")
        try:
            # 尝试连接
            print(f"  → 调用 rm_create_robot_arm('{ip}', {port})...")
            handle = arm.rm_create_robot_arm(ip, port)
            
            print(f"  → 返回 handle: {handle}")
            print(f"  → handle 类型: {type(handle)}")
            print(f"  → handle.id: {handle.id}")
            
            # 尝试获取软件版本
            print(f"  → 测试获取软件版本...")
            version_result = arm.rm_get_arm_software_info()
            print(f"  → 版本信息: {version_result}")
            
            # 尝试获取关节状态
            print(f"  → 测试获取关节状态...")
            joint_result = arm.rm_get_joint_degree()
            print(f"  → 关节状态: {joint_result}")
            
            print(f"✓ {name}: SDK 连接成功!")
            
        except Exception as e:
            print(f"✗ {name}: SDK 连接失败 - {e}")
            import traceback
            traceback.print_exc()
    
    print()
    print("=" * 60)
    print("诊断建议:")
    print("=" * 60)
    print("""
    如果 PING 失败:
      → 检查网线是否插好
      → 检查网络接口是否启用: ip addr show
      → 检查防火墙设置: sudo ufw status
    
    如果 PING 成功但 TCP 端口失败:
      → 检查机械臂是否开机
      → 确认机械臂 IP 配置是否正确
      → 检查端口号是否为 8080
    
    如果 TCP 端口成功但 SDK 连接失败:
      → 检查 SDK 版本是否匹配机械臂固件
      → 尝试重启机械臂
      → 检查是否有其他程序占用连接
    """)

if __name__ == "__main__":
    test_network()
