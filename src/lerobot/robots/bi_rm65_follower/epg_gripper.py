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
import json
from typing import Optional, Tuple, Dict, Any

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
        """检查是否已连接（临时连接模式下只检查初始化状态）"""
        return self._is_connected
    
    def connect(self) -> None:
        """连接到夹爪设备（仅初始化，不保持连接）"""
        # 不再保持长连接，只是标记为"已初始化"
        if self._is_connected:
            logger.debug("Gripper already initialized")
            return
            
        try:
            logger.info(f"Initializing gripper at {self.ip}:{self.port}...")
            # 临时连接用于初始化
            temp_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            temp_client.settimeout(5.0)
            temp_client.connect((self.ip, self.port))
            logger.info(f"Socket connected temporarily for initialization")
                
            time.sleep(0.5)
                
            # 初始化夹爪（使用临时客户端）
            logger.info("Initializing gripper device...")
            self._initialize_with_temp_client(temp_client)
            logger.info("Gripper device initialized successfully")
                
            # 关闭临时连接
            temp_client.close()
                
            # 标记为已初始化（但不保持连接）
            self._is_connected = True
            self.client = None  # 不保持客户端
                
        except socket.error as e:
            logger.error(f"Socket connection failed: {e}")
            logger.error(f"This may be caused by:")
            logger.error(f"  1. Port {self.port} already in use")
            logger.error(f"  2. Gripper device not powered on")
            logger.error(f"  3. Network connectivity issue")
            self._is_connected = False
            raise
        except Exception as e:
            logger.error(f"Failed to initialize gripper: {e}")
            self._is_connected = False
            raise
    
    def _recv_until_ok(
        self, 
        client: socket.socket, 
        expect_command: str, 
        timeout_s: float = 2.0
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        过滤 trajectory_state 推送消息，直到收到期望 command 的成功响应。
        
        Args:
            client: socket 客户端
            expect_command: 期望的命令名（如 "write_registers", "set_tool_voltage"）
            timeout_s: 超时时间（秒）
            
        Returns:
            (是否成功, 响应字典)
        """
        client.settimeout(0.2)
        deadline = time.time() + timeout_s
        last_msgs = []
        
        while time.time() < deadline:
            try:
                data = client.recv(4096)
                if not data:
                    continue
                text = data.decode(errors="ignore").strip()
                if not text:
                    continue
                
                # 有时会多条消息混在一起，按行分割
                for part in [p for p in text.splitlines() if p.strip()]:
                    last_msgs.append(part)
                    
                    # 丢弃轨迹状态推送
                    if "current_trajectory_state" in part:
                        logger.debug(f"Filtered out trajectory_state push")
                        continue
                    
                    try:
                        j = json.loads(part)
                    except Exception:
                        logger.debug(f"Failed to parse JSON: {part[:100]}")
                        continue
                    
                    # 检查是否是期望的命令响应
                    if j.get("command") != expect_command:
                        logger.debug(f"Skipping non-matching command: {j.get('command')}")
                        continue
                    
                    # 检查是否成功（多种可能的成功标志）
                    ok = (
                        j.get("write_state") is True
                        or j.get("state") is True
                        or j.get("set_state") is True
                    )
                    if ok:
                        logger.debug(f"Received OK response for {expect_command}")
                        return True, j
                    else:
                        logger.warning(f"Received non-OK response for {expect_command}: {j}")
                        return False, j
            
            except socket.timeout:
                continue
            except Exception as e:
                logger.error(f"Error in _recv_until_ok: {e}")
                return False, {"error": str(e), "last_msgs": last_msgs[-5:]}
        
        logger.warning(f"Timeout waiting for {expect_command} response")
        return False, {"error": "timeout", "last_msgs": last_msgs[-10:]}
    
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
        """初始化夹爪设备（不使能，保持自由活动）"""
        logger.info("Initializing gripper...")
        
        # 1. 设置工具电压为24V
        cmd = '{"command":"set_tool_voltage","voltage_type":3}\r\n'
        self._send_command(cmd)
        time.sleep(0.3)
        
        # 2. 配置Modbus模式
        cmd = '{"command":"set_modbus_mode","port":1,"baudrate":115200,"timeout":2}\r\n'
        self._send_command(cmd)
        time.sleep(0.3)
        
        # 3. 去使能（让夹爪可以自由活动）
        cmd = f'{{"command":"write_registers","port":1,"address":1000,"num":1,"data":[0,0],"device":{self.device_id}}}\r\n'
        self._send_command(cmd)
        time.sleep(0.3)
        
        logger.info("Gripper initialized in free-move mode (disabled, can be moved manually)")
    
    def _initialize_with_temp_client(self, client: socket.socket) -> None:
        """使用临时客户端初始化夹爪（用于临时连接模式）"""
        logger.info("Initializing gripper with temporary client...")
            
        # 1. 设置工具电压为24V
        cmd1 = '{"command":"set_tool_voltage","voltage_type":3}\r\n'
        client.sendall(cmd1.encode('utf-8'))
        ok, resp = self._recv_until_ok(client, "set_tool_voltage", timeout_s=2.0)
        if not ok:
            logger.warning(f"set_tool_voltage failed: {resp}")
        time.sleep(0.2)
            
        # 2. 配置 Modbus 模式
        cmd2 = '{"command":"set_modbus_mode","port":1,"baudrate":115200,"timeout":2}\r\n'
        client.sendall(cmd2.encode('utf-8'))
        ok, resp = self._recv_until_ok(client, "set_modbus_mode", timeout_s=2.0)
        if not ok:
            logger.warning(f"set_modbus_mode failed: {resp}")
        time.sleep(0.2)
            
        # 3. 去使能（让夹爪可以自由活动）
        cmd3 = f'{{"command":"write_registers","port":1,"address":1000,"num":1,"data":[0,0],"device":{self.device_id}}}\r\n'
        client.sendall(cmd3.encode('utf-8'))
        ok, resp = self._recv_until_ok(client, "write_registers", timeout_s=2.0)
        if not ok:
            logger.warning(f"write_registers (disable) failed: {resp}")
        time.sleep(0.2)
            
        logger.info("Gripper initialized in free-move mode")
    
    def enable(self, force: Optional[int] = None, speed: Optional[int] = None) -> bool:
        """
        使能夹爪（进入力控模式）
        
        Args:
            force: 力度 (0-255), 默认使用初始化时的值
            speed: 速度 (0-255), 默认使用初始化时的值
        
        Returns:
            是否使能成功
        """
        if not self.is_connected:
            logger.error("Gripper not connected")
            return False
        
        force = force if force is not None else self.force
        speed = speed if speed is not None else self.speed
        
        try:
            # 1. 设置力度和速度
            cmd = f'{{"command":"write_registers","port":1,"address":1002,"num":1,"data":[{force},{speed}],"device":{self.device_id}}}\r\n'
            self._send_command(cmd)
            time.sleep(0.1)
            
            # 2. 上使能
            cmd = f'{{"command":"write_registers","port":1,"address":1000,"num":1,"data":[0,1],"device":{self.device_id}}}\r\n'
            self._send_command(cmd)
            time.sleep(0.1)
            
            logger.info(f"Gripper enabled with force={force}, speed={speed}")
            return True
        except Exception as e:
            logger.error(f"Failed to enable gripper: {e}")
            return False
    
    def disable(self) -> bool:
        """
        去使能夹爪（让夹爪可以自由活动）
        
        Returns:
            是否去使能成功
        """
        if not self.is_connected:
            logger.error("Gripper not connected")
            return False
        
        try:
            cmd = f'{{"command":"write_registers","port":1,"address":1000,"num":1,"data":[0,0],"device":{self.device_id}}}\r\n'
            self._send_command(cmd)
            time.sleep(0.1)
            logger.info("Gripper disabled (free-move mode)")
            return True
        except Exception as e:
            logger.error(f"Failed to disable gripper: {e}")
            return False
    
    def set_position(self, position: float, blocking: bool = False) -> bool:
        """
        设置夹爪位置（临时连接模式）
        
        Args:
            position: 目标位置 (0-100, 0=完全张开, 100=完全闭合)
            blocking: 是否等待完成
            
        Returns:
            是否设置成功
        """
        # [GRIPDBG] 记录输入参数
        logger.warning(f"[GRIPDBG] set_position called position_0_100={position}")
        
        if not self.is_connected:
            logger.error("Gripper not initialized")
            return False
        
        # 归一化：0-100 -> 0-255
        pos_raw = int((position / 100.0) * 255)
        pos_raw = max(0, min(255, pos_raw))
        
        try:
            # 临时连接
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.settimeout(2.0)
            client.connect((self.ip, self.port))
            
            # 发送位置指令（设置目标位置和速度）
            cmd1 = f'{{"command":"write_registers","port":1,"address":1001,"num":1,"data":[{pos_raw},{self.speed}],"device":{self.device_id}}}\r\n'
            client.sendall(cmd1.encode('utf-8'))
            ok1, resp1 = self._recv_until_ok(client, "write_registers", timeout_s=2.0)
            # [GRIPDBG] 记录写 1001 结果
            logger.warning(f"[GRIPDBG] write 1001 ok={ok1} resp={resp1}")
            if not ok1:
                logger.warning(f"write_registers (address 1001) failed: {resp1}")
                client.close()
                return False
            
            # 发送使能指令（触发夹爪移动）
            cmd2 = f'{{"command":"write_registers","port":1,"address":1000,"num":1,"data":[0,9],"device":{self.device_id}}}\r\n'
            client.sendall(cmd2.encode('utf-8'))
            ok2, resp2 = self._recv_until_ok(client, "write_registers", timeout_s=2.0)
            # [GRIPDBG] 记录写 1000 结果
            logger.warning(f"[GRIPDBG] write 1000 [0,9] ok={ok2} resp={resp2}")
            if not ok2:
                logger.warning(f"write_registers (address 1000) failed: {resp2}")
                client.close()
                return False
            
            client.close()
            
            if blocking:
                time.sleep(1.0)  # 等待夹爪移动
            
            logger.debug(f"Set gripper to {position}% ({pos_raw}/255)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set gripper position: {e}")
            return False
    
    def get_position(self, skip_buffer_clear: bool = False) -> Optional[int]:
        """
        读取夹爪当前位置（临时连接模式）
        
        Args:
            skip_buffer_clear: 跳过缓冲区清空（提高性能但可能不稳定）
        
        Returns:
            当前位置 (0-255)，失败返回 None
        """
        if not self.is_connected:
            logger.error("Gripper not initialized")
            return None
        
        try:
            # 临时连接
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.settimeout(2.0)
            client.connect((self.ip, self.port))
            
            # 读取寄存器 1001 获取当前位置
            cmd = f'{{"command":"read_holding_registers","port":1,"address":1001,"num":1,"device":{self.device_id}}}\r\n'
            client.sendall(cmd.encode('utf-8'))
            
            # 使用消息过滤读取响应
            position = self._recv_read_response(client, timeout_s=2.0)
            client.close()
            
            if position is not None:
                logger.debug(f"Gripper position: {position}")
            else:
                logger.warning("Failed to read gripper position")
            
            return position
            
        except Exception as e:
            logger.error(f"Failed to read gripper position: {e}")
            return None
    
    def _recv_read_response(self, client: socket.socket, timeout_s: float = 2.0) -> Optional[int]:
        """
        过滤消息并读取 read_holding_registers 的响应。
        
        Args:
            client: socket 客户端
            timeout_s: 超时时间（秒）
            
        Returns:
            夹爪位置 (0-255)，失败返回 None
        """
        client.settimeout(0.2)
        deadline = time.time() + timeout_s
        
        while time.time() < deadline:
            try:
                data = client.recv(4096)
                if not data:
                    continue
                text = data.decode(errors="ignore").strip()
                if not text:
                    continue
                
                # 按行分割处理多条消息
                for part in [p for p in text.splitlines() if p.strip()]:
                    # 丢弃轨迹状态推送
                    if "current_trajectory_state" in part:
                        logger.debug(f"Filtered out trajectory_state push")
                        continue
                    
                    try:
                        j = json.loads(part)
                    except Exception:
                        logger.debug(f"Failed to parse JSON: {part[:100]}")
                        continue
                    
                    # 查找 read_holding_registers 的响应
                    cmd_name = j.get('command', '')
                    if 'read' in cmd_name and 'holding_registers' in cmd_name:
                        if 'data' in j:
                            data_field = j['data']
                            
                            # 解析位置数据
                            if isinstance(data_field, list) and len(data_field) > 0:
                                position = data_field[0]
                                return position
                            elif isinstance(data_field, int):
                                position = (data_field >> 8) & 0xFF
                                return position
            
            except socket.timeout:
                continue
            except Exception as e:
                logger.error(f"Error in _recv_read_response: {e}")
                return None
        
        logger.warning(f"Timeout waiting for read_holding_registers response")
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


# 别名，保持与其他代码的兼容性
EPGGripper = EPGGripperClient
