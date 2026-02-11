#!/usr/bin/env python3
"""
RM65 推理脚本 - 基于官方 record.py 逻辑
结合官方 predict_action 函数与 RM65 特有功能（ACK等待、Delta控制）
"""

import argparse
import logging
import time
from pathlib import Path
from copy import copy

import numpy as np
import torch
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.bi_rm65_follower import BiRM65Follower
from lerobot.robots.bi_rm65_follower.config_bi_rm65_follower import BiRM65FollowerConfig
from lerobot.utils.control_utils import predict_action

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="RM65 推理脚本（官方逻辑风格）")
    
    # 模型相关
    parser.add_argument("--pretrained-policy-path", type=str, required=True,
                        help="预训练模型路径（本地目录）")
    parser.add_argument("--device", type=str, default="cuda",
                        help="推理设备：cuda 或 cpu")
    
    # 机器人配置
    parser.add_argument("--left-arm-ip", type=str, default="169.254.128.20",
                        help="左臂IP地址")
    parser.add_argument("--right-arm-ip", type=str, default="169.254.128.21",
                        help="右臂IP地址")
    
    # 任务配置
    parser.add_argument("--task", type=str, 
                        default="Pick up the white part and place it in the corresponding position of the gold part",
                        help="任务描述（必须与训练时完全一致）")
    parser.add_argument("--fps", type=float, default=6.0,
                        help="推理频率（Hz）")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="推理持续时间（秒）")
    
    # 控制参数
    parser.add_argument("--max-delta-per-step", type=float, default=5.0,
                        help="单步最大角度变化（度，安全限制）")
    parser.add_argument("--delta-scale", type=float, default=1.0,
                        help="Delta放大系数（解决移动幅度过小问题）")
    parser.add_argument("--profile", action="store_true",
                        help="开启性能分析（输出各步骤耗时）")
    
    args = parser.parse_args()
    
    # ========== 1. 加载模型和预处理器 ==========
    logger.info(f"加载模型: {args.pretrained_policy_path}")
    policy_path = Path(args.pretrained_policy_path)
    
    if not policy_path.exists():
        raise FileNotFoundError(f"模型路径不存在: {policy_path}")
    
    # 加载策略（官方方式）
    policy = PreTrainedPolicy.from_pretrained(str(policy_path))
    policy.to(args.device)
    policy.eval()
    
    # 获取数据集统计信息（用于归一化）
    # 从训练输出目录找到对应的数据集
    dataset_repo_id = "joyandai/lerobot_v3"  # 替换为你的数据集ID
    logger.info(f"加载数据集统计: {dataset_repo_id}")
    dataset = LeRobotDataset(dataset_repo_id)
    
    # 创建预处理器和后处理器（官方方式）
    logger.info("初始化预处理器和后处理器")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(policy_path),
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": args.device}
        }
    )
    
    # ========== 2. 初始化机器人 ==========
    logger.info("初始化 RM65 机器人")
    robot_config = BiRM65FollowerConfig(
        left_arm_ip=args.left_arm_ip,
        right_arm_ip=args.right_arm_ip
    )
    robot = BiRM65Follower(robot_config)
    robot.connect()
    
    logger.info("机器人连接成功，获取初始位置")
    obs = robot.get_observation()
    
    # 解析初始位置（13维状态）
    state13 = np.array([obs[key] for key in sorted(obs.keys()) if ".pos" in key], dtype=np.float32)
    logger.info(f"初始状态: {state13}")
    
    # IMPORTANT: 模型输出是 Delta（增量），需要累积到当前位置
    current_positions = state13.copy()  # 用于累积 delta
    logger.info("注意：SmolVLA 输出的是 Delta，将累积到当前位置")
    
    # ========== 3. 推理循环 ==========
    logger.info(f"开始推理（{args.fps} Hz，持续 {args.duration} 秒）")
    logger.info(f"任务: {args.task}")
    
    dt = 1.0 / args.fps
    start_time = time.time()
    tick = 0
    
    try:
        while time.time() - start_time < args.duration:
            loop_start = time.time()
            
            # 3.1 获取观测
            t1 = time.time()
            obs = robot.get_observation()
            t_obs = time.time() - t1
            
            # 构建观测帧（官方格式）
            observation_frame = {}
            for key, value in obs.items():
                if "image" in key:
                    # 图像: HWC uint8 -> numpy array
                    observation_frame[key] = value
                else:
                    # 状态: float
                    observation_frame[key] = np.array(value, dtype=np.float32)
            
            # 3.2 推理（官方 predict_action 函数）
            t2 = time.time()
            with torch.inference_mode():
                action = predict_action(
                    observation=observation_frame,
                    policy=policy,
                    device=torch.device(args.device),
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    use_amp=(args.device == "cuda"),
                    task=args.task,
                    robot_type="bi_rm65_follower"
                )
            t_infer = time.time() - t2
            
            # action 是 torch.Tensor，shape [13]，经过 postprocessor 后是反归一化的 Delta
            action_np = action.cpu().numpy()
            
            # Delta 放大（解决移动幅度过小问题）
            if args.delta_scale != 1.0:
                action_np = action_np * args.delta_scale
            
            # 安全裁剪：限制单步最大 Delta（防止危险运动）
            action_np_clipped = np.clip(action_np, -args.max_delta_per_step, args.max_delta_per_step)
            
            # IMPORTANT: 累积 Delta 到当前位置（SmolVLA 输出的是增量）
            current_positions += action_np_clipped  # delta 累积
            
            # 3.3 构建动作字典（使用累积后的位置）
            action_names = dataset.features["action"]["names"]
            action_to_send = {name: float(current_positions[i]) for i, name in enumerate(action_names)}
            
            # 3.4 直接发送动作
            t3 = time.time()
            robot.send_action(action_to_send)
            t_send = time.time() - t3
            
            # 3.5 日志输出
            loop_time = time.time() - loop_start
            if args.profile:
                logger.info(f"[{tick:04d}] Obs:{t_obs*1000:.0f}ms Infer:{t_infer*1000:.0f}ms Send:{t_send*1000:.0f}ms Total:{loop_time*1000:.0f}ms FPS:{1/loop_time:.1f}")
            elif tick % 10 == 0:
                logger.info(f"[{tick:04d}] Delta: {action_np_clipped.round(2)} Target: {current_positions.round(1)}")
            
            tick += 1
            
            # 3.9 控制频率
            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
            elif args.profile and elapsed > dt * 1.2:
                logger.warning(f"[{tick:04d}] 循环超时: {elapsed*1000:.0f}ms > {dt*1000:.0f}ms")
    
    except KeyboardInterrupt:
        logger.info("用户中断推理")
    finally:
        logger.info("断开机器人连接")
        robot.disconnect()
        logger.info(f"推理完成，共执行 {tick} 步")


if __name__ == "__main__":
    main()
