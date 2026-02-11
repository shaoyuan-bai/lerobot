#!/usr/bin/env python3
"""
测试夹爪输出值（只加载模型和夹爪，不控制机械臂）

使用方法：
python test_gripper_output_only.py

功能：
1. 加载训练好的模型
2. 连接夹爪（不连接机械臂）
3. 实时显示：
   - Policy 原始输出（归一化后的值）
   - 反归一化后的值（应该是 0-100）
   - 发送给硬件的值（0-255）
   - 夹爪实际位置
"""

import time
import numpy as np
import torch
from pathlib import Path
from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.robots.configs import RobotConfig
from lerobot.robots.bi_rm65_follower.epg_gripper import EPGGripperClient
from lerobot.common.utils.utils import init_hydra_config
from transformers import PreTrainedConfig

# ==================== 配置 ====================
MODEL_PATH = "/home/wooshrobot/bai/lerobot/outputs/train/rm65_smolvla_gripper_test_v2/checkpoints/040000/pretrained_model"
GRIPPER_IP = "192.168.1.18"  # 修改为你的夹爪IP
GRIPPER_PORT = 8080
DEVICE_ID = 1

# ==================== 加载模型 ====================
print("=" * 80)
print("🤖 加载模型...")
print("=" * 80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载 policy 配置
policy_cfg = PreTrainedConfig.from_pretrained(MODEL_PATH)
print(f"Policy 类型: {policy_cfg.name}")

# 加载 policy
policy = make_policy(
    policy_cfg=policy_cfg,
    pretrained_path=MODEL_PATH,
    device=device
)
policy.eval()

print("✅ 模型加载完成\n")

# 检查归一化统计
print("=" * 80)
print("📊 检查模型的归一化统计...")
print("=" * 80)

# 从 postprocessor 提取反归一化器
unnormalizer = None
if hasattr(policy, 'postprocessor'):
    for step in policy.postprocessor.steps:
        if step.__class__.__name__ == "UnnormalizerProcessorStep":
            unnormalizer = step
            break

if unnormalizer and hasattr(unnormalizer, 'stats'):
    if 'action' in unnormalizer.stats:
        action_stats = unnormalizer.stats['action']
        print(f"Action 统计信息:")
        print(f"  Mean: {action_stats['mean']}")
        print(f"  Std:  {action_stats['std']}")
        
        # 检查夹爪维度
        if len(action_stats['mean']) >= 13:
            print(f"\n夹爪维度 (索引12):")
            print(f"  Mean: {action_stats['mean'][12]:.4f}")
            print(f"  Std:  {action_stats['std'][12]:.4f}")
else:
    print("⚠️  未找到反归一化统计信息")

print()

# ==================== 连接夹爪 ====================
print("=" * 80)
print("🤏 连接夹爪...")
print("=" * 80)

gripper = EPGGripperClient(
    ip=GRIPPER_IP,
    port=GRIPPER_PORT,
    device_id=DEVICE_ID,
    force=100,
    speed=100
)

try:
    gripper.connect()
    print("✅ 夹爪连接成功\n")
except Exception as e:
    print(f"❌ 夹爪连接失败: {e}")
    print("提示: 请检查夹爪IP和端口是否正确")
    exit(1)

# ==================== 模拟推理循环 ====================
print("=" * 80)
print("🔄 开始监控夹爪输出（按 Ctrl+C 停止）")
print("=" * 80)
print()
print(f"{'时间':>8} | {'Policy原始':>12} | {'反归一化':>12} | {'发送值(0-255)':>15} | {'实际位置':>12} | {'状态':>8}")
print("-" * 95)

step_count = 0
start_time = time.time()

try:
    while True:
        step_count += 1
        elapsed = time.time() - start_time
        
        # ========== 1. 模拟 Policy 输出 ==========
        # 注意：这里我们需要模拟一个完整的 observation 输入
        # 实际推理时，这个 observation 来自相机和机械臂
        # 为了测试，我们只关注 action 输出
        
        # 创建虚拟观测（因为我们没有连接相机）
        # 这里只是为了让模型能运行，实际值不重要
        with torch.no_grad():
            # 注意：实际的 observation 结构取决于你的 policy 类型
            # 这里假设是 SmolVLA，可能需要图像输入
            
            # 为了简化，我们直接读取模型内部状态
            # 如果需要真实推理，需要提供真实的相机输入
            
            # ========== 简化方案：直接测试反归一化 ==========
            # 创建一个测试的归一化 action（13维）
            # 我们只关注第12维（夹爪）
            
            # 模拟一个归一化的 action（Policy 输出）
            # 正常情况下，归一化后的值应该在 [-3, 3] 范围内
            # 但如果统计错误，可能会很奇怪
            
            # 我们测试几个典型值
            test_values = [
                -2.0,  # 很小的值
                -1.0,  # 小值
                0.0,   # 中间值
                1.0,   # 大值
                2.0,   # 很大的值
            ]
            
            # 循环测试不同值
            test_idx = (step_count - 1) % len(test_values)
            normalized_gripper = test_values[test_idx]
            
            # 创建完整的 13 维 action（只有夹爪维度是测试值，其他随机）
            normalized_action = torch.randn(1, 13, device=device)
            normalized_action[0, 12] = normalized_gripper  # 夹爪是第12维
            
            # ========== 2. 反归一化 ==========
            if unnormalizer:
                # 使用模型的反归一化器
                action_dict = {"action": normalized_action}
                unnormalized_dict = unnormalizer(action_dict)
                unnormalized_action = unnormalized_dict["action"][0].cpu().numpy()
                gripper_unnormalized = unnormalized_action[12]
            else:
                # 手动反归一化（如果没有 unnormalizer）
                mean = action_stats['mean'][12]
                std = action_stats['std'][12]
                gripper_unnormalized = normalized_gripper * std + mean
            
            # ========== 3. 限制范围并转换为 0-255 ==========
            # 这是 rm65_follower.py 中的逻辑
            gripper_0_100 = float(gripper_unnormalized)
            gripper_0_100 = max(0, min(100, gripper_0_100))  # 限制到 0-100
            
            # 转换为 0-255
            gripper_0_255 = int((gripper_0_100 / 100.0) * 255)
            gripper_0_255 = max(0, min(255, gripper_0_255))
            
            # ========== 4. 发送给夹爪 ==========
            success = gripper.set_position(gripper_0_100, blocking=False)
            
            # ========== 5. 读取实际位置 ==========
            actual_pos = gripper.get_position(skip_buffer_clear=True)
            if actual_pos is None:
                actual_pos_str = "N/A"
            else:
                actual_pos_str = f"{actual_pos}"
            
            # ========== 6. 状态判断 ==========
            if gripper_unnormalized < 0:
                status = "⚠️负值"
            elif gripper_unnormalized > 100:
                status = "⚠️超限"
            elif success:
                status = "✅正常"
            else:
                status = "❌失败"
            
            # ========== 7. 打印输出 ==========
            print(f"{elapsed:>7.1f}s | {normalized_gripper:>11.4f} | {gripper_unnormalized:>11.4f} | {gripper_0_255:>15d} | {actual_pos_str:>12} | {status:>8}")
        
        # 每个测试值停留 2 秒
        time.sleep(2.0)
        
        # 测试完一轮后，显示分隔线
        if test_idx == len(test_values) - 1:
            print("-" * 95)

except KeyboardInterrupt:
    print("\n\n⏹️  停止测试")

finally:
    # ==================== 清理 ====================
    print("\n" + "=" * 80)
    print("🧹 清理资源...")
    print("=" * 80)
    
    gripper.disconnect()
    print("✅ 夹爪已断开")
    
    print("\n📊 测试统计:")
    print(f"  总运行时间: {time.time() - start_time:.1f}s")
    print(f"  总步数: {step_count}")
    print()
    print("💡 注意事项:")
    print("  1. 如果'反归一化'列出现负值，说明归一化统计有问题")
    print("  2. '发送值(0-255)'被限制在 0-255 范围内")
    print("  3. 如果夹爪不动，检查'反归一化'列是否都是0或很小的值")
    print()
