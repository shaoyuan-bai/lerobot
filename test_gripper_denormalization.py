#!/usr/bin/env python3
"""
测试夹爪反归一化逻辑（不需要相机，只需要模型和夹爪）

使用方法：
python test_gripper_denormalization.py

功能：
1. 加载模型，提取归一化统计
2. 连接夹爪
3. 测试不同的归一化值，观察反归一化后的结果和实际夹爪行为
"""

import time
import numpy as np
import torch
import sys
from pathlib import Path
from transformers import PreTrainedConfig

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lerobot.robots.bi_rm65_follower.epg_gripper import EPGGripperClient
from lerobot.policies.factory import make_pre_post_processors

# ==================== 配置 ====================
MODEL_PATH = "/home/wooshrobot/bai/lerobot/outputs/train/rm65_smolvla_gripper_test_v2/checkpoints/040000/pretrained_model"
GRIPPER_IP = "192.168.1.18"  # 修改为你的夹爪IP
GRIPPER_PORT = 8080
DEVICE_ID = 1

def extract_unnormalizer_stats(model_path):
    """从模型中提取反归一化统计（直接读取配置文件）"""
    try:
        import json
        from pathlib import Path
        
        model_path = Path(model_path)
        
        # 方法1：读取 policy_postprocessor.json（LeRobot 标准位置）
        postprocessor_config_path = model_path / "policy_postprocessor.json"
        if postprocessor_config_path.exists():
            print("  → 从 policy_postprocessor.json 读取...")
            with open(postprocessor_config_path, 'r') as f:
                config = json.load(f)
            
            # 调试：打印配置结构
            print(f"  → 配置顶层键: {list(config.keys())}")
            if 'steps' in config:
                print(f"  → 找到 {len(config['steps'])} 个 processor steps")
                for i, step in enumerate(config['steps']):
                    # 可能的键名：type, class, name, _target_, __class__
                    step_type = step.get('type') or step.get('class') or step.get('name') or step.get('_target_') or step.get('__class__', 'unknown')
                    print(f"     Step {i}: {step_type}")
                    print(f"       → Step 的所有键: {list(step.keys())}")
                    
                    # 检查是否是 unnormalizer（各种可能的命名）
                    step_str = str(step).lower()
                    if 'unnormalizer' in step_str or 'unnormaliz' in step_type.lower():
                        print(f"       → 找到 unnormalizer step!")
                        print(f"       → 完整内容: {json.dumps(step, indent=10)[:500]}")
            
            # 查找 unnormalizer_processor 的 stats
            for step in config.get('steps', []):
                registry_name = step.get('registry_name', '')
                
                if registry_name == 'unnormalizer_processor':
                    print("  ✅ 找到 unnormalizer_processor step")
                    
                    # 方法1：从 state_file 读取
                    if 'state_file' in step:
                        state_file = model_path / step['state_file']
                        print(f"  → 尝试从 state_file 读取: {step['state_file']}")
                        
                        if state_file.exists():
                            import torch
                            
                            # 判断文件格式
                            if state_file.suffix == '.safetensors':
                                # 使用 safetensors 读取
                                try:
                                    from safetensors import safe_open
                                    state = {}
                                    with safe_open(state_file, framework="pt", device="cpu") as f:
                                        for key in f.keys():
                                            state[key] = f.get_tensor(key)
                                    print(f"  → 使用 safetensors 读取成功")
                                except ImportError:
                                    print(f"  ⚠️  需要安装 safetensors: pip install safetensors")
                                    state = None
                            else:
                                # 使用 torch.load 读取（添加 weights_only=False）
                                state = torch.load(state_file, map_location='cpu', weights_only=False)
                            
                            if state:
                                print(f"  → state 的键: {list(state.keys())}")
                                
                                if 'stats' in state:
                                    stats = state['stats']
                                    if 'action' in stats:
                                        print("  ✅ 成功从 state_file 提取 action 统计")
                                        return {
                                            'action': {
                                                'mean': stats['action']['mean'],
                                                'std': stats['action']['std']
                                            }
                                        }
                                else:
                                    # stats 可能直接是 tensor
                                    # 尝试从 state 中提取 mean 和 std
                                    if 'action.mean' in state and 'action.std' in state:
                                        print("  ✅ 从 state 直接提取 action.mean 和 action.std")
                                        return {
                                            'action': {
                                                'mean': state['action.mean'],
                                                'std': state['action.std']
                                            }
                                        }
                                    else:
                                        print(f"  ⚠️  state 中没有 stats，可用的键: {list(state.keys())}")
                        else:
                            print(f"  ⚠️  state_file 不存在: {state_file}")
                    
                    # 方法2：从 config 中读取（如果 stats 直接存在 config 里）
                    if 'config' in step:
                        step_config = step['config']
                        if 'stats' in step_config:
                            stats = step_config['stats']
                            if 'action' in stats:
                                print("  ✅ 成功从 config 提取 action 统计")
                                import torch
                                return {
                                    'action': {
                                        'mean': torch.tensor(stats['action']['mean']),
                                        'std': torch.tensor(stats['action']['std'])
                                    }
                                }
                    
                    print(f"  ⚠️  找到 unnormalizer 但无法提取 stats")
                    print(f"       step 的键: {list(step.keys())}")

        
        # 方法3：读取 config.json 中的 dataset_stats
        config_path = model_path / "config.json"
        if config_path.exists():
            print("  → 从 config.json 读取...")
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # 有些模型把 stats 存在 config 里
            if 'dataset_stats' in config:
                stats = config['dataset_stats']
                import torch
                return {
                    'action': {
                        'mean': torch.tensor(stats['action']['mean']),
                        'std': torch.tensor(stats['action']['std'])
                    }
                }
        
        print("  ⚠️  未找到归一化统计")
        print("  → 尝试列出模型目录文件...")
        if model_path.exists():
            files = list(model_path.glob('*.json'))
            print(f"  → 找到 {len(files)} 个 JSON 文件:")
            for f in files:
                print(f"     - {f.name}")
        
        return None
        
    except Exception as e:
        print(f"提取统计失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def denormalize_action(normalized_value, mean, std):
    """手动反归一化"""
    return normalized_value * std + mean

def main():
    print("=" * 80)
    print("🔍 测试夹爪反归一化逻辑")
    print("=" * 80)
    print()
    
    # ========== 1. 提取归一化统计 ==========
    print("📊 从模型中提取归一化统计...")
    try:
        stats = extract_unnormalizer_stats(MODEL_PATH)
        if stats is None or 'action' not in stats:
            print("❌ 无法提取归一化统计")
            return
        
        action_mean = stats['action']['mean']
        action_std = stats['action']['std']
        
        print(f"✅ 成功提取统计信息")
        print(f"Action 维度: {len(action_mean)}")
        
        if len(action_mean) < 13:
            print(f"❌ Action 维度不足 13，实际: {len(action_mean)}")
            return
        
        gripper_mean = action_mean[12]
        gripper_std = action_std[12]
        
        print()
        print("夹爪维度 (索引12) 统计:")
        print(f"  Mean: {gripper_mean:.4f}")
        print(f"  Std:  {gripper_std:.4f}")
        print()
        
    except Exception as e:
        print(f"❌ 提取统计失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========== 2. 连接夹爪 ==========
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
        print("✅ 夹爪连接成功")
        print()
    except Exception as e:
        print(f"❌ 夹爪连接失败: {e}")
        print("提示: 请检查夹爪IP和网络连接")
        return
    
    # ========== 3. 测试不同的归一化值 ==========
    print("=" * 80)
    print("🧪 测试反归一化（每个值持续 3 秒）")
    print("=" * 80)
    print()
    
    # 测试值：从小到大
    test_cases = [
        ("很小的负值", -3.0),
        ("小负值", -2.0),
        ("负值", -1.0),
        ("接近零", -0.5),
        ("零", 0.0),
        ("小正值", 0.5),
        ("正值", 1.0),
        ("大正值", 2.0),
        ("很大的正值", 3.0),
    ]
    
    print(f"{'描述':<15} | {'归一化值':>10} | {'反归一化':>12} | {'限制后':>10} | {'发送(0-255)':>12} | {'状态':>8}")
    print("-" * 85)
    
    try:
        for desc, normalized_val in test_cases:
            # 反归一化
            denormalized = denormalize_action(normalized_val, gripper_mean, gripper_std)
            
            # 限制到 0-100
            clamped = max(0.0, min(100.0, denormalized))
            
            # 转换为 0-255
            value_255 = int((clamped / 100.0) * 255)
            
            # 判断状态
            if denormalized < 0:
                status = "⚠️负值"
            elif denormalized > 100:
                status = "⚠️超限"
            else:
                status = "✅正常"
            
            # 打印
            print(f"{desc:<15} | {normalized_val:>10.2f} | {denormalized:>12.4f} | {clamped:>10.2f} | {value_255:>12d} | {status:>8}")
            
            # 发送给夹爪
            success = gripper.set_position(clamped, blocking=False)
            if not success:
                print(f"  ❌ 发送失败")
            
            # 等待 3 秒，观察夹爪
            time.sleep(3.0)
            
            # 读取实际位置
            actual_pos = gripper.get_position(skip_buffer_clear=True)
            if actual_pos is not None:
                print(f"  → 实际位置: {actual_pos}/255")
            
            print()
    
    except KeyboardInterrupt:
        print("\n⏹️  测试被中断")
    
    finally:
        # ========== 4. 清理 ==========
        print("=" * 80)
        print("🧹 清理资源...")
        print("=" * 80)
        gripper.disconnect()
        print("✅ 夹爪已断开")
    
    # ========== 5. 分析结果 ==========
    print()
    print("=" * 80)
    print("📋 分析结果")
    print("=" * 80)
    print()
    
    print("关键观察点:")
    print()
    print("1️⃣  如果'反归一化'列出现大量负值:")
    print("   → 说明归一化统计错误（Mean 和 Std 不匹配数据集）")
    print("   → 需要重新训练，使用正确的数据集统计")
    print()
    print("2️⃣  如果'限制后'列都是 0 或很小的值:")
    print("   → 夹爪会保持张开状态（position=0）")
    print("   → 这就是你看到的'夹爪不动'或'只会张开'的原因")
    print()
    print("3️⃣  正常情况下:")
    print("   → 归一化值在 [-3, 3] 范围")
    print("   → 反归一化后应该在 [0, 100] 范围")
    print("   → 夹爪应该有明显的开合动作")
    print()
    print("4️⃣  当前统计:")
    print(f"   → Gripper Mean: {gripper_mean:.4f}")
    print(f"   → Gripper Std:  {gripper_std:.4f}")
    print()
    print("   如果是 RM65 数据集，期望值约为:")
    print("      Mean ≈ 9.39")
    print("      Std  ≈ 13.33")
    print()
    print("   如果是 SO100 预训练模型的统计:")
    print("      Mean ≈ 12.00")
    print("      Std  ≈ 19.04")
    print()

if __name__ == "__main__":
    main()
