#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
钧舵EPG夹爪控制模块
通过Socket连接到RM65机械臂控制器，使用Modbus协议控制夹爪
"""

import logging
import socket
import time
from typing import Optional

logger = logging.getLogger(__name__)


class EPGGripperClient:
    """
    EPG夹爪客户端
    
    通过Socket连接到机械臂控制器，使用JSON格式的Modbus命令控制夹爪。
    位置范围：0-255 (0=完全张开，255=完全闭合)
    """
    
    def __init__(
        self,
        ip: str = "169.254.128.21",
        port: int = 8080,
        device_id: int = 9,
        force: int = 60,
        speed: int = 255,
    ):
        """
        初始化夹爪客户端
        
        Args:
            ip: 机械臂控制器IP地址
            port: 通信端口
            device_id: 夹爪设备ID
            force: 默认夹持力度 (0-255)
            speed: 默认运动速度 (0-255)
        """
        self.ip = ip
        self.port = port
        self.device_id = device_id
        self.force = force
        self.speed = speed
        self.client: Optional[socket.socket] = None
        self._is_connected = False
        
    @property
    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self._is_connected and self.client is not None
    
    def connect(self) -> None:
        """连接到夹爪设备"""
        if self.is_connected:
            logger.warning("Gripper already connected")
            return
        
        try:
            logger.info(f"Attempting to connect gripper to {self.ip}:{self.port}...")
            self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client.settimeout(5.0)  # 5秒超时
            
            # 尝试连接
            self.client.connect((self.ip, self.port))
            logger.info(f"Socket connected to {self.ip}:{self.port}")
            time.sleep(0.5)
            
            # 初始化夹爪
            logger.info("Initializing gripper device...")
            self._initialize()
            self._is_connected = True
            logger.info("Gripper device initialized successfully")
            
        except socket.error as e:
            logger.error(f"Socket connection failed: {e}")
            logger.error(f"This may be caused by:")
            logger.error(f"  1. Port {self.port} already in use by RM65 SDK")
            logger.error(f"  2. Gripper device not powered on")
            logger.error(f"  3. Network connectivity issue")
            if self.client:
                self.client.close()
                self.client = None
            raise
        except Exception as e:
            logger.error(f"Failed to initialize gripper: {e}")
            if self.client:
                self.client.close()
                self.client = None
            raise
    
    def _send_command(self, command: str) -> bool:
        """
        发送命令到夹爪
        
        Args:
            command: JSON格式的命令字符串
            
        Returns:
            是否发送成功
        """
        if not self.is_connected:
            logger.error("Gripper not connected")
            return False
        
        try:
            self.client.send(command.encode('utf-8'))
            time.sleep(0.1)  # 短暂延迟确保命令执行
            return True
        except Exception as e:
            logger.error(f"Failed to send command: {e}")
            return False
    
    def _initialize(self) -> None:
        """初始化夹爪设备"""
        logger.info("Initializing gripper...")
        
        # 1. 设置工具电压为24V
        cmd = '{"command":"set_tool_voltage","voltage_type":3}\r\n'
        self._send_command(cmd)
        time.sleep(0.5)
        
        # 2. 配置Modbus模式
        cmd = '{"command":"set_modbus_mode","port":1,"baudrate":115200,"timeout":2}\r\n'
        self._send_command(cmd)
        time.sleep(0.5)
        
        # 3. 去使能（复位）
        cmd = f'{{"command":"write_registers","port":1,"address":1000,"num":1,"data":[0,0],"device":{self.device_id}}}\r\n'
        self._send_command(cmd)
        time.sleep(0.5)
        
        # 4. 上使能
        cmd = f'{{"command":"write_registers","port":1,"address":1000,"num":1,"data":[0,1],"device":{self.device_id}}}\r\n'
        self._send_command(cmd)
        time.sleep(0.5)
        
        # 5. 设置默认力度和速度
        cmd = f'{{"command":"write_registers","port":1,"address":1002,"num":1,"data":[{self.force},{self.speed}],"device":{self.device_id}}}\r\n'
        self._send_command(cmd)
        time.sleep(0.5)
        
        logger.info("Gripper initialized successfully")
    
    def set_position(self, position: int) -> bool:
        """
        设置夹爪位置
        
        Args:
            position: 目标位置 (0-255, 0=完全张开, 255=完全闭合)
            
        Returns:
            是否设置成功
        """
        if not self.is_connected:
            logger.error("Gripper not connected")
            return False
        
        # 确保位置在有效范围内
        position = max(0, min(255, int(position)))
        
        try:
            # 1. 设置目标位置
            cmd = f'{{"command":"write_registers","port":1,"address":1001,"num":1,"data":[{position},{self.speed}],"device":{self.device_id}}}\r\n'
            if not self._send_command(cmd):
                return False
            
            # 2. 触发执行
            cmd = f'{{"command":"write_registers","port":1,"address":1000,"num":1,"data":[0,9],"device":{self.device_id}}}\r\n'
            return self._send_command(cmd)
            
        except Exception as e:
            logger.error(f"Failed to set gripper position: {e}")
            return False
    
    def get_position(self) -> Optional[int]:
        """
        读取夹爪当前位置
        
        Returns:
            当前位置 (0-255)，失败返回 None
        """
        if not self.is_connected:
            logger.error("Gripper not connected")
            return None
        
        try:
            # 清空接收缓冲区，避免读到历史响应
            self.client.setblocking(False)
            try:
                while True:
                    self.client.recv(1024)
            except BlockingIOError:
                pass  # 缓冲区已清空
            finally:
                self.client.setblocking(True)
            
            # 读取寄存器1001获取当前位置
            cmd = f'{{"command":"read_holding_registers","port":1,"address":1001,"num":1,"device":{self.device_id}}}\r\n'
            self.client.send(cmd.encode('utf-8'))
            time.sleep(0.1)  # 等待响应
            
            # 接收响应
            response = self.client.recv(1024).decode('utf-8').strip()
            
            # 解析JSON响应
            # 响应格式: {"command":"read_holding_registers","data":25855}
            # data是16位数值，高字节=位置，低字节=速度
            import json
            for line in response.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                    if data.get('command') == 'read_holding_registers' and 'data' in data:
                        # 提取16位数据
                        combined_value = data['data']
                        # 高字节 = 位置 (0-255)
                        position = (combined_value >> 8) & 0xFF
                        # 低字节 = 速度 (0-255)
                        speed = combined_value & 0xFF
                        logger.debug(f"Gripper position: {position}, speed: {speed}")
                        return position
                except json.JSONDecodeError:
                    continue
            
            logger.warning("Failed to parse gripper position from response")
            return None
            
        except Exception as e:
            logger.error(f"Failed to read gripper position: {e}")
            return None
    
    def disconnect(self) -> None:
        """断开夹爪连接"""
        if self.client:
            try:
                self.client.close()
                logger.info("Gripper disconnected")
            except Exception as e:
                logger.error(f"Error disconnecting gripper: {e}")
            finally:
                self.client = None
                self._is_connected = False
