#!/usr/bin/env python

"""
调试夹爪Socket响应格式
用于查看read_holding_registers命令的返回数据格式
"""

import socket
import time
import json

def test_read_position():
    """测试读取夹爪位置并打印原始响应"""
    
    ip = "169.254.128.21"
    port = 8080
    device_id = 9
    
    print("连接到夹爪...")
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.settimeout(5.0)
    client.connect((ip, port))
    print(f"✓ 已连接到 {ip}:{port}\n")
    
    # 初始化夹爪
    print("初始化夹爪...")
    commands = [
        '{"command":"set_tool_voltage","voltage_type":3}\r\n',
        '{"command":"set_modbus_mode","port":1,"baudrate":115200,"timeout":2}\r\n',
        f'{{"command":"write_registers","port":1,"address":1000,"num":1,"data":[0,0],"device":{device_id}}}\r\n',
        f'{{"command":"write_registers","port":1,"address":1000,"num":1,"data":[0,1],"device":{device_id}}}\r\n',
        f'{{"command":"write_registers","port":1,"address":1002,"num":1,"data":[60,255],"device":{device_id}}}\r\n',
    ]
    
    for cmd in commands:
        client.send(cmd.encode('utf-8'))
        time.sleep(0.3)
    
    print("✓ 初始化完成\n")
    
    # 先设置一个已知位置
    print("设置夹爪到位置 100...")
    cmd = f'{{"command":"write_registers","port":1,"address":1001,"num":1,"data":[100,255],"device":{device_id}}}\r\n'
    client.send(cmd.encode('utf-8'))
    time.sleep(0.2)
    
    cmd = f'{{"command":"write_registers","port":1,"address":1000,"num":1,"data":[0,9],"device":{device_id}}}\r\n'
    client.send(cmd.encode('utf-8'))
    time.sleep(2)  # 等待运动完成
    
    # 读取位置
    print("\n" + "="*60)
    print("测试读取夹爪位置")
    print("="*60)
    
    for i in range(3):
        print(f"\n尝试 {i+1}:")
        
        # 发送读取命令
        cmd = f'{{"command":"read_holding_registers","port":1,"address":1001,"num":1,"device":{device_id}}}\r\n'
        print(f"发送命令: {cmd.strip()}")
        client.send(cmd.encode('utf-8'))
        
        # 接收响应
        try:
            response = client.recv(1024)
            print(f"原始响应 (bytes): {response}")
            print(f"解码响应 (str): {response.decode('utf-8', errors='ignore')}")
            
            # 尝试解析JSON
            try:
                response_str = response.decode('utf-8').strip()
                # 可能有多行，分别解析
                for line in response_str.split('\n'):
                    if line.strip():
                        print(f"解析行: {line.strip()}")
                        data = json.loads(line.strip())
                        print(f"JSON数据: {json.dumps(data, indent=2)}")
            except json.JSONDecodeError as e:
                print(f"JSON解析失败: {e}")
        except socket.timeout:
            print("超时，未收到响应")
        
        time.sleep(1)
    
    # 测试其他寄存器
    print("\n" + "="*60)
    print("测试读取其他寄存器")
    print("="*60)
    
    registers = [
        (1000, "控制寄存器"),
        (1001, "位置寄存器"),
        (1002, "力度/速度寄存器"),
    ]
    
    for addr, desc in registers:
        print(f"\n读取地址 {addr} ({desc}):")
        cmd = f'{{"command":"read_holding_registers","port":1,"address":{addr},"num":1,"device":{device_id}}}\r\n'
        print(f"命令: {cmd.strip()}")
        client.send(cmd.encode('utf-8'))
        time.sleep(0.2)
        
        try:
            response = client.recv(1024)
            print(f"响应: {response.decode('utf-8', errors='ignore')}")
        except socket.timeout:
            print("超时")
    
    # 断开连接
    print("\n断开连接...")
    client.close()
    print("✓ 测试完成")


if __name__ == "__main__":
    try:
        test_read_position()
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
