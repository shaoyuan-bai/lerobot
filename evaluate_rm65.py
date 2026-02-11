#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

from lerobot.cameras.ffmpeg import FFmpegCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.processor.converters import observation_to_transition, transition_to_batch
from lerobot.processor.core import TransitionKey
from lerobot.robots.bi_rm65_follower.config_bi_rm65_follower import BiRM65FollowerConfig
from lerobot.robots.bi_rm65_follower.bi_rm65_follower import BiRM65Follower
from lerobot.utils.utils import init_logging, log_say


def npimg_to_torch_chw01(img: np.ndarray) -> torch.Tensor:
    if not img.flags["WRITEABLE"]:
        img = img.copy()
    return torch.from_numpy(img).permute(2, 0, 1).contiguous().float() / 255.0  # (C,H,W)


def resize_chw(img_chw: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
    x = img_chw.unsqueeze(0)  # (1,C,H,W)
    x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
    return x.squeeze(0)  # (C,H,W)


def get_current_joints_6(robot_obs: Dict, side: str) -> np.ndarray:
    assert side in ("left", "right")
    keys = [f"{side}_joint_{i}.pos" for i in range(1, 7)]
    return np.array([float(robot_obs[k]) for k in keys], dtype=np.float32)


def arm_action_from_target6(target6: np.ndarray) -> Dict[str, float]:
    # RM65Follower 期望 joint_1.pos ... joint_6.pos（没有 left_/right_ 前缀）
    return {f"joint_{i}.pos": float(target6[i - 1]) for i in range(1, 7)}


def extract_policy_expected_image_keys(policy) -> List[str]:
    cfg = getattr(policy, "config", None)
    if cfg is None:
        return []
    img_feats = getattr(cfg, "image_features", None)
    if isinstance(img_feats, dict):
        return list(img_feats.keys())
    return []


def fill_policy_image_keys_in_batch(batch: Dict, k_fixed: str, k_handeye: str) -> None:
    if k_fixed in batch and k_handeye in batch:
        return
    candidates = [k for k in ("observation.image", "observation.image2", "observation.image3") if k in batch]
    if len(candidates) >= 2:
        batch.setdefault(k_fixed, batch[candidates[0]])
        batch.setdefault(k_handeye, batch[candidates[1]])
        return
    # 退化：能补一个就补一个
    if k_fixed not in batch and k_handeye in batch:
        batch[k_fixed] = batch[k_handeye]
    if k_handeye not in batch and k_fixed in batch:
        batch[k_handeye] = batch[k_fixed]


def parse_args():
    p = argparse.ArgumentParser("RM65 SmolVLA evaluation v3 (stable timing)")

    p.add_argument("--policy-path", type=str, required=True)
    p.add_argument("--task", type=str, required=True)

    p.add_argument("--num-episodes", type=int, default=1)
    p.add_argument("--episode-time-s", type=int, default=60)

    p.add_argument("--cam-fps", type=int, default=10)
    p.add_argument("--target-fps", type=int, default=8)

    p.add_argument("--left-arm-ip", type=str, default="169.254.128.20")
    p.add_argument("--right-arm-ip", type=str, default="169.254.128.21")

    p.add_argument("--control-arm", choices=["left", "right"], default="right")
    p.add_argument("--delta-scale", type=float, default=5.0)
    p.add_argument("--delta-clip", type=float, default=5.0)

    p.add_argument("--cam-width", type=int, default=640)
    p.add_argument("--cam-height", type=int, default=360)
    p.add_argument("--model-image-size", type=int, default=256)

    p.add_argument("--send-left", action="store_true", help="也发送左臂动作（默认不发）")
    p.add_argument("--send-every", type=int, default=1, help="每 N 帧发送一次动作（默认 1）")

    p.add_argument("--no-display", action="store_true")
    p.add_argument("--no-sound", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    init_logging()
    logging.getLogger("lerobot.robots").setLevel(logging.INFO)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"🚀 Using device: {device}")

    # ✅ 修复：严格使用 args.cam_width / args.cam_height
    camera_config = {
        "top": FFmpegCameraConfig(
            index_or_path="/dev/video0",
            width=args.cam_width,
            height=args.cam_height,
            fps=args.cam_fps,
        ),
        "wrist": FFmpegCameraConfig(
            index_or_path="/dev/video2",
            width=args.cam_width,
            height=args.cam_height,
            fps=args.cam_fps,
        ),
    }

    robot_config = BiRM65FollowerConfig(
        left_arm_ip=args.left_arm_ip,
        right_arm_ip=args.right_arm_ip,
        cameras=camera_config,
        id="rm65_follower",
    )

    logging.info("正在初始化 RM65 机器人...")
    robot = BiRM65Follower(robot_config)

    logging.info(f"正在加载策略: {args.policy_path}")
    cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(args.policy_path).to(device).eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.policy_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    policy_img_keys = extract_policy_expected_image_keys(policy)
    preferred_fixed = policy_img_keys[1] if len(policy_img_keys) > 1 else "observation.images.fixed"
    preferred_handeye = policy_img_keys[0] if len(policy_img_keys) > 0 else "observation.images.handeye"

    logging.info(f"Loaded policy type: {cfg.type}")
    logging.info(f"Loaded policy class: {policy.__class__}")
    logging.info(f"Expected policy image feature keys (from policy.config.image_features): {policy_img_keys}")

    logging.info("正在连接机器人...")
    robot.connect()
    if not robot.is_connected:
        raise RuntimeError("机器人连接失败！")

    dt = 1.0 / max(1, args.target_fps)

    logging.info(f"开始执行策略推理，共 {args.num_episodes} 个回合")
    logging.info("按 Ctrl+C 可随时停止")
    logging.info(
        f"✅ Delta ON | delta-scale={args.delta_scale}, delta-clip={args.delta_clip}, control-arm={args.control_arm} | "
        f"target-fps={args.target_fps} | cam-fps={args.cam_fps} | send-left={args.send_left} | send-every={args.send_every}"
    )

    try:
        for epi in range(args.num_episodes):
            log_say(f"执行第 {epi+1}/{args.num_episodes} 个回合", play_sounds=not args.no_sound)
            policy.reset()

            start_wall = time.time()
            start = time.perf_counter()
            next_tick = start
            frame = 0

            while time.time() - start_wall < args.episode_time_s:
                loop_start = time.perf_counter()

                # -------- obs --------
                t0 = time.perf_counter()
                robot_obs = robot.get_observation()
                t1 = time.perf_counter()

                cur_left6 = get_current_joints_6(robot_obs, "left")
                cur_right6 = get_current_joints_6(robot_obs, "right")

                top_chw = npimg_to_torch_chw01(robot_obs["top"])      # (C,H,W) 640x360
                wrist_chw = npimg_to_torch_chw01(robot_obs["wrist"])  # (C,H,W) 640x360

                out_hw = (args.model_image_size, args.model_image_size)
                top_256 = resize_chw(top_chw, out_hw)
                wrist_256 = resize_chw(wrist_chw, out_hw)

                obs: Dict[str, torch.Tensor] = {}

                state6 = cur_right6 if args.control_arm == "right" else cur_left6
                obs["observation.state"] = torch.from_numpy(state6).float().unsqueeze(0)  # (1,6)

                # policy 期望的 keys：给 640x360 的原图，且 BCHW
                obs[preferred_fixed] = top_chw.unsqueeze(0)      # (1,3,H,W)
                obs[preferred_handeye] = wrist_chw.unsqueeze(0)  # (1,3,H,W)

                # 兼容 normalizer/preprocessor
                obs["observation.image"] = top_256.unsqueeze(0)
                obs["observation.image2"] = wrist_256.unsqueeze(0)
                obs["observation.image3"] = top_256.unsqueeze(0)

                transition = observation_to_transition(obs)
                transition[TransitionKey.COMPLEMENTARY_DATA] = {"task": args.task}

                # -------- infer --------
                with torch.inference_mode():
                    processed_transition = preprocessor._forward(transition)
                    batch = transition_to_batch(processed_transition)
                    fill_policy_image_keys_in_batch(batch, preferred_fixed, preferred_handeye)

                    # 保证所有图像是 BCHW
                    for k, v in list(batch.items()):
                        if isinstance(v, torch.Tensor) and v.ndim == 3:
                            batch[k] = v.unsqueeze(0)

                    action_raw = policy.select_action(batch)
                    action_proc = postprocessor(action_raw)

                t2 = time.perf_counter()

                # -------- parse delta6 --------
                if isinstance(action_proc, torch.Tensor):
                    act = action_proc
                elif isinstance(action_proc, dict) and "action" in action_proc and isinstance(action_proc["action"], torch.Tensor):
                    act = action_proc["action"]
                else:
                    raise ValueError(f"Unexpected processed_action: {type(action_proc)}")

                act_np = act.squeeze(0).float().cpu().numpy()
                delta6 = act_np[:6].astype(np.float32)

                if args.delta_clip and args.delta_clip > 0:
                    delta6 = np.clip(delta6, -float(args.delta_clip), float(args.delta_clip))
                delta6 = delta6 * float(args.delta_scale)

                if frame == 0:
                    logging.info(f"First delta6 used: {delta6}")
                    logging.info(f"Current right joints: {cur_right6}")

                if args.control_arm == "right":
                    tgt_right6 = cur_right6 + delta6
                    tgt_left6 = cur_left6
                else:
                    tgt_left6 = cur_left6 + delta6
                    tgt_right6 = cur_right6

                # -------- act --------
                send_now = (frame % max(1, args.send_every) == 0)
                if send_now:
                    if args.control_arm == "right":
                        robot.right_arm.send_action(arm_action_from_target6(tgt_right6))
                        if args.send_left:
                            robot.left_arm.send_action(arm_action_from_target6(tgt_left6))
                    else:
                        robot.left_arm.send_action(arm_action_from_target6(tgt_left6))
                        if args.send_left:
                            robot.right_arm.send_action(arm_action_from_target6(tgt_right6))

                t3 = time.perf_counter()

                frame += 1

                obs_ms = (t1 - t0) * 1000
                infer_ms = (t2 - t1) * 1000
                act_ms = (t3 - t2) * 1000
                total_ms = (t3 - t0) * 1000

                # -------- fixed tick scheduling --------
                next_tick += dt
                now = time.perf_counter()
                sleep_t = next_tick - now
                if sleep_t > 0:
                    time.sleep(sleep_t)
                    overrun_ms = 0.0
                else:
                    # 这一帧超时了（不 sleep），但下一帧仍按 next_tick 推进，不“越跑越慢”
                    overrun_ms = (-sleep_t) * 1000

                if frame % 50 == 0:
                    sleep_ms = max(0.0, sleep_t) * 1000
                    logging.info(
                        f"帧 {frame}: obs={obs_ms:.1f}ms, infer={infer_ms:.1f}ms, act={act_ms:.1f}ms, "
                        f"total={total_ms:.1f}ms | sleep={sleep_ms:.1f}ms | overrun={overrun_ms:.1f}ms "
                        f"(目标周期: {dt*1000:.1f}ms)"
                    )

            elapsed = time.perf_counter() - start
            actual_fps = frame / max(1e-6, elapsed)
            logging.info(f"回合 {epi+1} 完成，实际帧率: {actual_fps:.2f} fps")

    except KeyboardInterrupt:
        logging.info("收到中断信号，正在停止...")

    finally:
        logging.info("正在断开机器人连接...")
        robot.disconnect()
        log_say("推理完成", play_sounds=not args.no_sound)


if __name__ == "__main__":
    main()
