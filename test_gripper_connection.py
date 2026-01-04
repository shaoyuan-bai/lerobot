#!/usr/bin/env python
"""
测试EPG夹爪连接
直接测试Socket连接到169.254.128.21:8080
"""

import socket
import time
import sys

def test_socket_connection():
    """测试基础Socket连接"""
    ip = "169.254.128.21"
    port = 8080
    
    print(f"=" * 60)
    print(f"测试Socket连接到 {ip}:{port}")
    print(f"=" * 60)
    
    try:
        print(f"\n1. 创建Socket...")
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(5.0)
        print(f"   ✓ Socket创建成功")
        
        print(f"\n2. 尝试连接...")
        client.connect((ip, port))
        print(f"   ✓ 连接成功!")
        
        print(f"\n3. 发送测试命令...")
        # 读取夹爪位置
        cmd = '{"command":"read_holding_registers","port":1,"address":1001,"num":1,"device":9}\r\n'
        client.send(cmd.encode('utf-8'))
        print(f"   ✓ 命令已发送")
        
        print(f"\n4. 等待响应...")
        time.sleep(0.2)
        response = client.recv(1024).decode('utf-8')
        print(f"   ✓ 收到响应: {response[:100]}")
        
        print(f"\n5. 关闭连接...")
        client.close()
        print(f"   ✓ 连接已关闭")
        
        print(f"\n" + "=" * 60)
        print(f"✅ 测试成功! 夹爪通信正常")
        print(f"=" * 60)
        return True
        
    except socket.timeout:
        print(f"\n❌ 连接超时!")
        print(f"   可能原因:")
        print(f"   - 设备未开机")
        print(f"   - IP地址错误")
        print(f"   - 网络不通")
        return False
        
    except ConnectionRefusedError:
        print(f"\n❌ 连接被拒绝!")
        print(f"   可能原因:")
        print(f"   - 端口{port}未开放")
        print(f"   - 服务未启动")
        return False
        
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"\n❌ 端口被占用!")
            print(f"   错误: {e}")
            print(f"   可能原因:")
            print(f"   - RM65 SDK正在使用端口{port}")
            print(f"   - 其他程序占用了此端口")
            print(f"\n   解决方案:")
            print(f"   1. 检查是否已连接RM65 (可能SDK占用了端口)")
            print(f"   2. 使用 'netstat -tuln | grep {port}' 查看端口占用")
            print(f"   3. 尝试不同的连接顺序")
        else:
            print(f"\n❌ Socket错误: {e}")
        return False
        
    except Exception as e:
        print(f"\n❌ 未知错误: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_rm_sdk():
    """测试在RM SDK连接后能否连接夹爪"""
    print(f"\n" + "=" * 60)
    print(f"测试场景: RM SDK + 夹爪同时连接")
    print(f"=" * 60)
    
    try:
        # 导入RM SDK
        from Robotic_Arm.rm_robot_interface import RoboticArm
        
        print(f"\n1. 创建RM SDK连接...")
        arm = RoboticArm()
        handle = arm.rm_create_robot_arm("169.254.128.21", 8080)
        
        if handle.id == -1:
            print(f"   ⚠ RM SDK连接失败 (handle.id=-1)")
        else:
            print(f"   ✓ RM SDK已连接 (handle.id={handle.id})")
        
        print(f"\n2. 尝试创建夹爪Socket连接...")
        success = test_socket_connection()
        
        print(f"\n3. 断开RM SDK...")
        arm.rm_delete_robot_arm()
        print(f"   ✓ RM SDK已断开")
        
        return success
        
    except ImportError:
        print(f"   ❌ 无法导入RM SDK")
        print(f"   跳过此测试")
        return None
    except Exception as e:
        print(f"   ❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("EPG夹爪连接诊断工具")
    print("=" * 60)
    
    # 测试1: 直接连接
    print(f"\n【测试1】直接Socket连接测试")
    result1 = test_socket_connection()
    
    # 测试2: 与RM SDK共存
    print(f"\n\n【测试2】与RM SDK共存测试")
    result2 = test_with_rm_sdk()
    
    # 总结
    print(f"\n\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"直接连接: {'✅ 成功' if result1 else '❌ 失败'}")
    print(f"与RM SDK共存: {'✅ 成功' if result2 else '❌ 失败' if result2 is False else '⊘ 未测试'}")
    
    if not result1:
        print(f"\n建议:")
        print(f"1. 检查机械臂是否开机并连接到网络")
        print(f"2. 运行 'ping 169.254.128.21' 测试网络连通性")
        print(f"3. 检查夹爪是否正确连接到机械臂控制器")
        print(f"4. 查看机械臂Web界面 http://169.254.128.21 确认设备状态")
    
    sys.exit(0 if result1 else 1)
