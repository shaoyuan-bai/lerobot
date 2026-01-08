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
RM65 双臂机器人推理脚本

使用训练好的策略控制 RM65 机器人执行任务。

用法示例:
    python evaluate_rm65.py \
        --policy-path /home/woosh/bai/lerobot/outputs/train/rm65_pickup_xuebi1/checkpoints/last/pretrained_model \
        --num-episodes 5 \
        --fps 20
"""

import argparse
import logging
import time

import numpy as np
import torch

from lerobot.cameras.ffmpeg import FFmpegCameraConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import make_default_robot_action_processor
from lerobot.processor.converters import observation_to_transition, transition_to_batch
from lerobot.processor.core import TransitionKey
from lerobot.robots.bi_rm65_follower.config_bi_rm65_follower import BiRM65FollowerConfig
from lerobot.robots.bi_rm65_follower.bi_rm65_follower import BiRM65Follower
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import _init_rerun


def parse_args():
    parser = argparse.ArgumentParser(description="RM65 双臂机器人策略推理")
    parser.add_argument(
        "--policy-path",
        type=str,
        required=True,
        help="训练好的模型路径（通常是 checkpoints/last/pretrained_model）",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=5,
        help="执行的回合数量",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="控制频率（帧每秒）",
    )
    parser.add_argument(
        "--episode-time-s",
        type=int,
        default=60,
        help="每个回合的最大时长（秒）",
    )
    parser.add_argument(
        "--left-arm-ip",
        type=str,
        default="169.254.128.20",
        help="左臂 IP 地址",
    )
    parser.add_argument(
        "--right-arm-ip",
        type=str,
        default="169.254.128.21",
        help="右臂 IP 地址",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="禁用 Rerun 可视化",
    )
    parser.add_argument(
        "--no-sound",
        action="store_true",
        help="禁用语音提示",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    init_logging()
    
    # 启用 DEBUG 级别日志
    logging.getLogger("lerobot.robots").setLevel(logging.DEBUG)

    # 创建相机配置（使用 FFmpeg 获取更高性能）
    camera_config = {
        "top": FFmpegCameraConfig(
            index_or_path="/dev/video0",
            width=1920,
            height=1080,
            fps=args.fps,
        ),
        "wrist": FFmpegCameraConfig(
            index_or_path="/dev/video2",
            width=1920,
            height=1080,
            fps=args.fps,
        ),
    }

    # 创建机器人配置
    robot_config = BiRM65FollowerConfig(
        left_arm_ip=args.left_arm_ip,
        right_arm_ip=args.right_arm_ip,
        cameras=camera_config,
        id="rm65_follower",
    )

    # 实例化机器人
    logging.info("正在初始化 RM65 机器人...")
    robot = BiRM65Follower(robot_config)

    # 加载训练好的策略，确保使用 GPU
    logging.info(f"正在加载策略: {args.policy_path}")
    from lerobot.policies.act.modeling_act import ACTPolicy
    
    # 确定设备（不依赖 policy.config.device）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"🚀 Using device: {device}")
    
    # 加载模型并移到指定设备
    policy = ACTPolicy.from_pretrained(args.policy_path).to(device).eval()

    # 创建预处理和后处理器（使用确定的设备）
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.policy_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    
    # DEBUG: 打印 preprocessor 的结构
    logging.info(f"Preprocessor type: {type(preprocessor)}")
    logging.info(f"Preprocessor: {preprocessor}")
    if hasattr(preprocessor, 'steps'):
        logging.info(f"Preprocessor steps: {preprocessor.steps}")

    # 创建机器人动作处理器
    robot_action_processor = make_default_robot_action_processor()

    # 连接机器人
    logging.info("正在连接机器人...")
    robot.connect()

    if not robot.is_connected:
        raise RuntimeError("机器人连接失败！")

    # 初始化键盘监听和可视化
    listener, events = init_keyboard_listener()
    if not args.no_display:
        _init_rerun(session_name="rm65_evaluation")

    logging.info(f"开始执行策略推理，共 {args.num_episodes} 个回合")
    logging.info("按 Ctrl+C 可随时停止")

    try:
        for episode_idx in range(args.num_episodes):
            log_say(f"执行第 {episode_idx + 1}/{args.num_episodes} 个回合", play_sounds=not args.no_sound)

            # 重置策略状态
            policy.reset()

            # 执行一个回合
            start_time = time.time()
            frame_count = 0

            while time.time() - start_time < args.episode_time_s:
                loop_start = time.perf_counter()

                # 获取机器人观测
                t0 = time.perf_counter()
                robot_obs = robot.get_observation()
                t1 = time.perf_counter()
                
                # DEBUG: 打印观测数据结构
                if frame_count == 0:
                    logging.info(f"Robot observation keys: {list(robot_obs.keys())}")
                    for key, value in robot_obs.items():
                        if isinstance(value, (torch.Tensor, np.ndarray)):
                            logging.info(f"  {key}: shape={getattr(value, 'shape', 'N/A')}, dtype={getattr(value, 'dtype', type(value))}")
                
                # 转换为策略期望的格式（不添加 batch 维度，让 preprocessor 处理）
                # 1. 合并关节位置为 state 向量 (13维: 12个关节 + 1个夹爪，但夹爪总是0)
                joint_keys = [
                    'left_joint_1.pos', 'left_joint_2.pos', 'left_joint_3.pos',
                    'left_joint_4.pos', 'left_joint_5.pos', 'left_joint_6.pos',
                    'right_joint_1.pos', 'right_joint_2.pos', 'right_joint_3.pos',
                    'right_joint_4.pos', 'right_joint_5.pos', 'right_joint_6.pos',
                ]
                state_values = [robot_obs[key] for key in joint_keys]
                state_values.append(0.0)  # 夹爪值，总是0
                state = np.array(state_values, dtype=np.float32)
                
                # 2. 重命名和转换图像（添加 "observation." 前缀以匹配策略期望）
                observation = {}
                observation['observation.state'] = torch.from_numpy(state)  # (13,)
                
                # 图像需要从 (H, W, C) 转为 (C, H, W)，并添加 "observation." 前缀
                # 同时转换为 float32 并归一化到 [0, 1] 以匹配训练时的格式
                for robot_key, obs_key in [('top', 'observation.images.top'), ('wrist', 'observation.images.wrist')]:
                    img = robot_obs[robot_key]  # (480, 640, 3) uint8
                    if isinstance(img, np.ndarray):
                        img = torch.from_numpy(img)
                    # 转换为 (C, H, W) 并转为 float32，范围 [0, 1]
                    img = img.permute(2, 0, 1).float() / 255.0  # (3, 480, 640) float32 in [0, 1]
                    observation[obs_key] = img

                # DEBUG: 打印 observation 键
                if frame_count == 0:
                    logging.info(f"Policy observation keys: {list(observation.keys())}")
                    for key, value in observation.items():
                        logging.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")

                # 转换为 transition 格式
                transition = observation_to_transition(observation)
                
                # DEBUG: 打印 transition 结构
                if frame_count == 0:
                    logging.info(f"Transition keys: {list(transition.keys())}")
                    obs_in_transition = transition.get(TransitionKey.OBSERVATION)
                    if obs_in_transition:
                        logging.info(f"Transition observation keys: {list(obs_in_transition.keys())}")

                # 处理观测并推理动作
                with torch.inference_mode():
                    # 最后一次检查 transition 结构
                    if frame_count == 0:
                        logging.info(f"Before preprocessor - transition type: {type(transition)}")
                        logging.info(f"Before preprocessor - has OBSERVATION key: {TransitionKey.OBSERVATION in transition}")
                        obs_check = transition.get(TransitionKey.OBSERVATION)
                        logging.info(f"Before preprocessor - observation value: {obs_check is not None and isinstance(obs_check, dict)}")
                        # 测试 copy() 行为
                        test_copy = transition.copy()
                        logging.info(f"After copy - has OBSERVATION key: {TransitionKey.OBSERVATION in test_copy}")
                        logging.info(f"After copy - observation value: {test_copy.get(TransitionKey.OBSERVATION) is not None}")
                    
                    try:
                        # 直接调用 _forward 而不是 __call__，因为 __call__ 会先调用 to_transition
                        # 而我们已经有了 transition 格式的数据
                        processed_transition = preprocessor._forward(transition)
                    except ValueError as e:
                        logging.error(f"Preprocessor failed: {e}")
                        logging.error(f"Transition keys at error: {list(transition.keys())}")
                        logging.error(f"Transition[OBSERVATION]: {transition.get(TransitionKey.OBSERVATION)}")
                        raise
                    
                    # DEBUG: 打印处理后的 observation 键
                    if frame_count == 0:
                        processed_obs = processed_transition.get(TransitionKey.OBSERVATION)
                        if processed_obs:
                            logging.info(f"After preprocessor - observation keys: {list(processed_obs.keys())}")
                    
                    # 转回 batch 格式以获取 observation
                    processed_batch = transition_to_batch(processed_transition)
                    
                    # DEBUG: 打印 batch 键
                    if frame_count == 0:
                        logging.info(f"After transition_to_batch - batch keys: {list(processed_batch.keys())}")
                    
                    action = policy.select_action(processed_batch)
                    processed_action = postprocessor(action)
                
                t2 = time.perf_counter()

                # 转换为机器人动作格式
                # processed_action 可能是 Tensor 或 dict
                if isinstance(processed_action, torch.Tensor):
                    # 如果是 Tensor，移除 batch 维度并转 numpy
                    action_array = processed_action.squeeze(0).cpu().numpy()  # (13,)
                    
                    # RM65 双臂机器人期望的动作格式：
                    # - 前 6 个值：左臂关节角度 (joint_1 到 joint_6)
                    # - 接下来 6 个值：右臂关节角度 (joint_1 到 joint_6)
                    # - 最后 1 个值：夹爪位置
                    robot_action = {}
                    
                    # 左臂动作 (joint_1 到 joint_6)
                    for i in range(6):
                        robot_action[f'left_joint_{i+1}.pos'] = float(action_array[i])
                    
                    # 右臂动作 (joint_1 到 joint_6)
                    for i in range(6):
                        robot_action[f'right_joint_{i+1}.pos'] = float(action_array[i+6])
                    
                    # 夹爪动作 (如果不是 0)
                    if len(action_array) > 12 and action_array[12] != 0:
                        robot_action['right_gripper.pos'] = float(action_array[12])
                    
                    # DEBUG: 打印动作字典
                    if frame_count == 0:
                        logging.info(f"Sending action with keys: {list(robot_action.keys())}")
                        logging.info(f"Action values (first 3): left=[{action_array[0]:.2f}, {action_array[1]:.2f}, {action_array[2]:.2f}], right=[{action_array[6]:.2f}, {action_array[7]:.2f}, {action_array[8]:.2f}]")
                    
                elif isinstance(processed_action, dict):
                    # 如果是 dict，移除每个值的 batch 维度
                    robot_action = {}
                    for key, value in processed_action.items():
                        if isinstance(value, torch.Tensor):
                            robot_action[key] = value.squeeze(0).cpu().numpy()
                        else:
                            robot_action[key] = value
                else:
                    raise ValueError(f"Unexpected processed_action type: {type(processed_action)}")

                # 发送动作到机器人
                robot.send_action(robot_action)
                t3 = time.perf_counter()

                # 帧率控制（修正负等待卡顿）
                frame_count += 1
                elapsed = time.perf_counter() - loop_start
                
                # 每 50 帧打印一次分段计时
                if frame_count % 50 == 0:
                    logging.info(
                        f"帧 {frame_count}: obs={((t1-t0)*1000):.1f}ms, "
                        f"infer={((t2-t1)*1000):.1f}ms, "
                        f"act={((t3-t2)*1000):.1f}ms, "
                        f"total={((t3-t0)*1000):.1f}ms (目标: {(1000/args.fps):.1f}ms)"
                    )
                
                # 避免"负等待"造成的卡顿
                dt = (1.0 / args.fps) - elapsed
                if dt > 0:
                    busy_wait(dt)

                # 检查是否需要提前退出
                if events.get("stop_recording", False):
                    log_say("收到停止信号", play_sounds=not args.no_sound)
                    break

            actual_fps = frame_count / (time.time() - start_time)
            logging.info(f"回合 {episode_idx + 1} 完成，实际帧率: {actual_fps:.2f} fps")

            # 回合间暂停
            if episode_idx < args.num_episodes - 1:
                log_say("准备下一个回合，请重置环境", play_sounds=not args.no_sound, blocking=True)
                time.sleep(2)

    except KeyboardInterrupt:
        logging.info("收到中断信号，正在停止...")

    finally:
        # 清理资源
        logging.info("正在断开机器人连接...")
        robot.disconnect()
        if listener is not None:
            listener.stop()
        log_say("推理完成", play_sounds=not args.no_sound)


if __name__ == "__main__":
    main()
