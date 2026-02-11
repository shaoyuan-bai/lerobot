#!/usr/bin/env python3
"""
验证 SmolVLA 模型输出的夹爪动作值

用法：
    python verify_smolvla_gripper_output.py

功能：
1. 加载训练好的 SmolVLA 模型
2. 从数据集中读取观测数据
3. 推理获取 action 输出
4. 分析夹爪维度（right_gripper.pos）的值分布
5. 检查是否有变化
"""

import sys
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors

# 配置
MODEL_PATH = "/home/wooshrobot/bai/lerobot/outputs/train/rm65_smolvla_pick/checkpoints/040000/pretrained_model"
DATASET_PATH = "C:/Users/ROG/.cache/huggingface/lerobot/joyandai/lerobot_v3_pick"  # Windows 路径
DATASET_REPO_ID = "joyandai/lerobot_v3_pick"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_SAMPLES = 50  # 测试样本数


def main():
    print("=" * 80)
    print("SmolVLA 模型夹爪输出验证工具")
    print("=" * 80)
    print()
    
    # 1. 检查 CUDA
    print(f"🔍 检查设备...")
    print(f"   PyTorch 版本: {torch.__version__}")
    print(f"   CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA 设备: {torch.cuda.get_device_name(0)}")
    print(f"   使用设备: {DEVICE}")
    print()
    
    # 2. 加载数据集
    print(f"📂 加载数据集: {DATASET_REPO_ID}")
    try:
        # 尝试从本地缓存加载
        dataset = LeRobotDataset(DATASET_REPO_ID)
        print(f"   ✅ 数据集加载成功")
        print(f"   总帧数: {len(dataset)}")
        print(f"   Episode 数: {dataset.num_episodes}")
    except Exception as e:
        print(f"   ❌ 数据集加载失败: {e}")
        print(f"   提示: 确保数据集已下载到本地")
        sys.exit(1)
    print()
    
    # 3. 加载模型
    print(f"🤖 加载模型: {MODEL_PATH}")
    try:
        # 加载策略
        from lerobot.configs.policies import PreTrainedConfig
        policy_cfg = PreTrainedConfig.from_pretrained(MODEL_PATH)
        policy_cfg.device = DEVICE
        
        policy = make_policy(policy_cfg, ds_meta=dataset.meta)
        policy.eval()
        
        # 加载预处理器
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=MODEL_PATH,
            dataset_stats=dataset.meta.stats,
            preprocessor_overrides={"device_processor": {"device": DEVICE}},
        )
        
        print(f"   ✅ 模型加载成功")
        print(f"   策略类型: {policy_cfg.type}")
        print(f"   输入特征: {list(policy_cfg.input_features.keys())}")
        print(f"   输出特征: {list(policy_cfg.output_features.keys())}")
    except Exception as e:
        print(f"   ❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    print()
    
    # 4. 推理并分析夹爪输出
    print(f"🔬 开始推理测试（{NUM_SAMPLES} 个样本）...")
    print()
    
    gripper_actions = []
    
    with torch.no_grad():
        for i in range(min(NUM_SAMPLES, len(dataset))):
            try:
                # 获取观测
                sample = dataset[i]
                
                # 预处理（过滤掉非tensor字段）
                tensor_sample = {
                    key: value.unsqueeze(0) 
                    for key, value in sample.items() 
                    if isinstance(value, torch.Tensor)
                }
                batch = preprocessor(tensor_sample)
                
                # 推理
                actions = policy.select_action(batch)
                
                # 后处理
                actions = postprocessor(actions)
                
                # 提取夹爪值（假设是最后一个维度）
                # action shape: (1, 13) -> [left_joints(6), right_joints(6), right_gripper(1)]
                action_array = actions["action"].cpu().numpy()[0]
                gripper_value = action_array[-1]  # 最后一个维度是夹爪
                
                gripper_actions.append(gripper_value)
                
                if i < 10 or i % 10 == 0:  # 显示前10个和每10个
                    print(f"   样本 {i:3d}: gripper={gripper_value:6.2f}")
                    
            except Exception as e:
                print(f"   ⚠️  样本 {i} 推理失败: {e}")
                continue
    
    print()
    print("=" * 80)
    print("📊 夹爪输出分析")
    print("=" * 80)
    
    if len(gripper_actions) == 0:
        print("❌ 没有成功的推理样本")
        sys.exit(1)
    
    gripper_actions = np.array(gripper_actions)
    
    print(f"✅ 成功推理 {len(gripper_actions)} 个样本")
    print()
    print(f"统计信息:")
    print(f"  最小值:     {gripper_actions.min():.2f}")
    print(f"  最大值:     {gripper_actions.max():.2f}")
    print(f"  平均值:     {gripper_actions.mean():.2f}")
    print(f"  标准差:     {gripper_actions.std():.2f}")
    print(f"  中位数:     {np.median(gripper_actions):.2f}")
    print()
    
    # 检查变化
    unique_values = len(np.unique(np.round(gripper_actions, 1)))
    print(f"唯一值数量（精度0.1）: {unique_values}")
    
    if unique_values <= 3:
        print()
        print("⚠️  警告: 夹爪输出值变化很小!")
        print("   模型可能没有学会控制夹爪")
        print("   建议:")
        print("   1. 检查训练数据中夹爪是否有足够的变化")
        print("   2. 增加训练数据量（更多 episodes）")
        print("   3. 延长训练时间（更多 steps）")
    else:
        print()
        print("✅ 夹爪输出有变化，模型可能学到了夹爪控制")
    
    print()
    print("夹爪值分布（前10个最常见的值）:")
    from collections import Counter
    counter = Counter([round(v, 1) for v in gripper_actions])
    for val, count in counter.most_common(10):
        percentage = count / len(gripper_actions) * 100
        print(f"  {val:6.1f}: {'█' * int(percentage / 2)} {count:3d} ({percentage:5.1f}%)")
    
    print()
    print("=" * 80)
    print("验证完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
