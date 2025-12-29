#!/usr/bin/env python3
"""测试 RM65 不同端口和连接方式"""

import socket
import requests
from Robotic_Arm.rm_robot_interface import RoboticArm

def test_ports():
    """测试不同端口"""
    ip = "169.254.128.20"
    common_ports = [8080, 8000, 80, 8888, 9090, 502, 503]
    
    print("=" * 60)
    print("测试 RM65 常用端口")
    print("=" * 60)
    print()
    
    for port in common_ports:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        try:
            result = sock.connect_ex((ip, port))
            if result == 0:
                print(f"✓ 端口 {port}: 可连接")
                
                # 尝试 HTTP GET
                try:
                    response = requests.get(f"http://{ip}:{port}/", timeout=2)
                    print(f"  → HTTP GET /: {response.status_code}")
                    if response.text:
                        print(f"  → 响应: {response.text[:100]}")
                except Exception as e:
                    print(f"  → HTTP GET 失败: {type(e).__name__}")
            else:
                print(f"✗ 端口 {port}: 不可连接")
        except Exception as e:
            print(f"✗ 端口 {port}: 异常 - {e}")
        finally:
            sock.close()
    
    print()

def test_sdk_modes():
    """测试 SDK 不同连接模式"""
    ip = "169.254.128.20"
    port = 8080
    
    print("=" * 60)
    print("测试 SDK 不同连接模式")
    print("=" * 60)
    print()
    
    # 测试1: 默认模式
    print("1️⃣  测试默认模式 RoboticArm()")
    try:
        arm = RoboticArm()
        handle = arm.rm_create_robot_arm(ip, port)
        print(f"   handle: {handle}")
        print(f"   handle.id: {handle.id}")
        
        if handle.id != -1:
            print("   ✓ 连接成功!")
            # 尝试获取状态
            ret = arm.rm_get_current_arm_state()
            print(f"   关节状态: {ret}")
            arm.rm_delete_robot_arm()
        else:
            print("   ✗ handle.id = -1, 连接失败")
    except Exception as e:
        print(f"   ✗ 异常: {e}")
    
    print()
    
    # 测试2: 检查是否有其他初始化参数
    print("2️⃣  检查 RoboticArm 类的参数")
    try:
        import inspect
        sig = inspect.signature(RoboticArm.__init__)
        print(f"   构造函数签名: {sig}")
        
        # 检查 rm_create_robot_arm 签名
        sig2 = inspect.signature(RoboticArm.rm_create_robot_arm)
        print(f"   rm_create_robot_arm 签名: {sig2}")
    except Exception as e:
        print(f"   ✗ 异常: {e}")
    
    print()

def test_api_endpoints():
    """测试可能的 API 端点"""
    ip = "169.254.128.20"
    port = 8080
    
    print("=" * 60)
    print("测试 HTTP API 端点")
    print("=" * 60)
    print()
    
    endpoints = [
        ("/", "GET"),
        ("/api", "GET"),
        ("/api/v1", "GET"),
        ("/initialize", "POST"),
        ("/init", "POST"),
        ("/robot/init", "POST"),
        ("/status", "GET"),
        ("/info", "GET"),
    ]
    
    for path, method in endpoints:
        try:
            url = f"http://{ip}:{port}{path}"
            if method == "GET":
                response = requests.get(url, timeout=2)
            else:
                response = requests.post(url, json={}, timeout=2)
            
            print(f"✓ {method:4} {path:20} → {response.status_code}")
            if response.text and len(response.text) < 200:
                print(f"  响应: {response.text}")
        except requests.exceptions.Timeout:
            print(f"⏱ {method:4} {path:20} → 超时")
        except requests.exceptions.ConnectionError as e:
            print(f"✗ {method:4} {path:20} → 连接失败")
        except Exception as e:
            print(f"✗ {method:4} {path:20} → {type(e).__name__}: {e}")
    
    print()

if __name__ == "__main__":
    test_ports()
    test_sdk_modes()
    test_api_endpoints()
    
    print("=" * 60)
    print("提示: 请查看睿尔曼 RM65 的官方文档或示例代码")
    print("确认正确的端口号和初始化步骤")
    print("=" * 60)
