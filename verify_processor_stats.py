#!/usr/bin/env python3
"""
验证脚本：检查训练后模型的归一化器是否使用了正确的数据集统计

用法：
    python verify_processor_stats.py --model_path /path/to/model --dataset_repo_id joyandai/lerobot_v3_pick
"""

import argparse
import torch
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to trained model checkpoint")
    parser.add_argument("--dataset_repo_id", type=str, required=True,
                        help="Dataset repo ID used for training")
    args = parser.parse_args()

    print(f"加载数据集: {args.dataset_repo_id}")
    dataset = LeRobotDataset(args.dataset_repo_id)
    
    print(f"加载模型配置: {args.model_path}")
    policy_cfg = PreTrainedConfig.from_pretrained(args.model_path)
    
    print("加载 processors...")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.model_path,
    )
    
    # 找到 normalizer 和 unnormalizer
    normalizer = None
    unnormalizer = None
    
    for step in preprocessor.steps:
        if step.__class__.__name__ == "NormalizerProcessorStep":
            normalizer = step
            break
    
    for step in postprocessor.steps:
        if step.__class__.__name__ == "UnnormalizerProcessorStep":
            unnormalizer = step
            break
    
    if normalizer is None:
        print("❌ 错误: 在 preprocessor 中未找到 NormalizerProcessorStep")
        return
    
    if unnormalizer is None:
        print("❌ 错误: 在 postprocessor 中未找到 UnnormalizerProcessorStep")
        return
    
    print("\n" + "="*80)
    print("验证归一化器统计")
    print("="*80)
    
    # 获取 action 维度
    action_key = "action"
    if action_key not in dataset.meta.stats:
        print(f"❌ 错误: 数据集统计中没有 '{action_key}' 键")
        return
    
    dataset_action_mean = dataset.meta.stats[action_key]["mean"]
    dataset_action_std = dataset.meta.stats[action_key]["std"]
    
    # 从模型的 normalizer 中获取
    if action_key in normalizer.stats:
        model_action_mean = normalizer.stats[action_key]["mean"]
        model_action_std = normalizer.stats[action_key]["std"]
    else:
        print(f"❌ 错误: 模型归一化器统计中没有 '{action_key}' 键")
        return
    
    print(f"\n数据集 action 统计:")
    print(f"  mean shape: {dataset_action_mean.shape}")
    print(f"  std shape:  {dataset_action_std.shape}")
    print(f"  mean: {dataset_action_mean}")
    print(f"  std:  {dataset_action_std}")
    
    print(f"\n模型归一化器 action 统计:")
    print(f"  mean shape: {np.array(model_action_mean).shape}")
    print(f"  std shape:  {np.array(model_action_std).shape}")
    print(f"  mean: {model_action_mean}")
    print(f"  std:  {model_action_std}")
    
    # 比较
    mean_match = np.allclose(dataset_action_mean, model_action_mean, rtol=1e-5)
    std_match = np.allclose(dataset_action_std, model_action_std, rtol=1e-5)
    
    print("\n" + "="*80)
    print("验证结果")
    print("="*80)
    
    if mean_match and std_match:
        print("✅ 成功! 模型归一化器使用了正确的数据集统计")
        print("   - action.mean 匹配 ✓")
        print("   - action.std 匹配 ✓")
    else:
        print("❌ 失败! 模型归一化器与数据集统计不匹配")
        if not mean_match:
            print("   - action.mean 不匹配 ✗")
            diff = np.abs(np.array(dataset_action_mean) - np.array(model_action_mean))
            print(f"     最大差异: {diff.max():.6f}")
        if not std_match:
            print("   - action.std 不匹配 ✗")
            diff = np.abs(np.array(dataset_action_std) - np.array(model_action_std))
            print(f"     最大差异: {diff.max():.6f}")
        
        print("\n⚠️  这意味着推理时会使用错误的反归一化,导致输出不正确!")
    
    # 检查夹爪维度(假设是最后一个维度)
    gripper_idx = len(dataset_action_mean) - 1
    print(f"\n夹爪维度检查 (index {gripper_idx}):")
    print(f"  数据集 - mean: {dataset_action_mean[gripper_idx]:.4f}, std: {dataset_action_std[gripper_idx]:.4f}")
    print(f"  模型   - mean: {model_action_mean[gripper_idx]:.4f}, std: {model_action_std[gripper_idx]:.4f}")
    
    if np.isclose(dataset_action_mean[gripper_idx], model_action_mean[gripper_idx], rtol=1e-5) and \
       np.isclose(dataset_action_std[gripper_idx], model_action_std[gripper_idx], rtol=1e-5):
        print("  ✅ 夹爪统计匹配")
    else:
        print("  ❌ 夹爪统计不匹配 - 这是导致夹爪不动的根本原因!")


if __name__ == "__main__":
    main()
