#!/usr/bin/env python
"""
RM65 双臂数据录制脚本

使用方法:
1. 运行: python record_rm65_demo.py
2. 按提示手动移动机械臂
3. 按回车开始/停止录制
4. 数据保存为 LeRobot 数据集格式

录制内容:
- 双臂关节角度 (12 个关节)
- 相机视频 (640×480@30fps)
- 时间戳同步
"""

import json
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2
from tqdm import tqdm

from lerobot.robots.bi_rm65_follower import BiRM65FollowerConfig, BiRM65Follower
from lerobot.cameras.opencv import OpenCVCameraConfig


class RM65DataRecorder:
    """RM65 数据录制器"""
    
    def __init__(self, output_dir="outputs/rm65_recordings"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 相机配置
        cameras_config = {
            "top": OpenCVCameraConfig(
                index_or_path=0,
                fps=30,
                width=640,
                height=480,
            ),
        }
        
        # 机器人配置
        self.config = BiRM65FollowerConfig(
            id="rm65_recorder",
            left_arm_ip="169.254.128.20",
            right_arm_ip="169.254.128.21",
            port=8080,
            move_speed=30,
            cameras=cameras_config,
        )
        
        self.robot = BiRM65Follower(self.config)
        self.episodes = []
    
    def connect(self):
        """连接机器人"""
        print("\n正在连接 RM65 双臂机器人...")
        self.robot.connect(calibrate=False)
        print("✓ 连接成功!")
    
    def disconnect(self):
        """断开连接"""
        if self.robot.is_connected:
            self.robot.disconnect()
            print("✓ 已断开连接")
    
    def record_episode(self, episode_id, duration=20, fps=30):
        """
        录制一个演示片段
        
        Args:
            episode_id: 片段编号
            duration: 录制时长 (秒)
            fps: 采样频率 (Hz)
        """
        print(f"\n" + "=" * 60)
        print(f"录制片段 #{episode_id}")
        print("=" * 60)
        
        # 准备阶段
        print("\n请将机械臂移动到起始位置...")
        input("按回车键开始录制...")
        
        print(f"\n🔴 开始录制 ({duration} 秒,{fps} Hz)")
        print("请演示任务...")
        
        # 录制数据
        frames = []
        interval = 1.0 / fps
        num_frames = int(duration * fps)
        
        start_time = time.time()
        
        for i in tqdm(range(num_frames), desc="录制中"):
            frame_start = time.time()
            
            # 读取观察数据 (关节 + 图像)
            obs = self.robot.get_observation()
            
            # 构建帧数据
            frame = {
                "timestamp": time.time() - start_time,
                "frame_index": i,
                # 关节角度
                "state": {k: v for k, v in obs.items() if k.endswith('.pos')},
                # 图像 (保存为路径,稍后写入)
                "images": {},
            }
            
            # 保存图像
            for cam_name in ["top"]:
                if cam_name in obs:
                    frame["images"][cam_name] = obs[cam_name]
            
            frames.append(frame)
            
            # 控制采样率
            elapsed = time.time() - frame_start
            if elapsed < interval:
                time.sleep(interval - elapsed)
        
        actual_duration = time.time() - start_time
        actual_fps = len(frames) / actual_duration
        
        print(f"\n✓ 录制完成!")
        print(f"  实际时长: {actual_duration:.2f}s")
        print(f"  实际帧率: {actual_fps:.1f} fps")
        print(f"  总帧数: {len(frames)}")
        
        # 保存片段
        self.save_episode(episode_id, frames)
        
        return frames
    
    def save_episode(self, episode_id, frames):
        """保存片段数据"""
        episode_dir = self.output_dir / f"episode_{episode_id:04d}"
        episode_dir.mkdir(exist_ok=True)
        
        print(f"\n正在保存片段 #{episode_id}...")
        
        # 保存元数据
        metadata = {
            "episode_id": episode_id,
            "num_frames": len(frames),
            "fps": 30,
            "duration": frames[-1]["timestamp"],
            "recorded_at": datetime.now().isoformat(),
        }
        
        with open(episode_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # 保存关节数据
        states = [frame["state"] for frame in frames]
        with open(episode_dir / "states.json", "w") as f:
            json.dump(states, f, indent=2)
        
        # 保存图像
        images_dir = episode_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        for i, frame in enumerate(tqdm(frames, desc="保存图像")):
            for cam_name, img in frame["images"].items():
                if isinstance(img, np.ndarray):
                    img_path = images_dir / f"{cam_name}_{i:06d}.jpg"
                    cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        print(f"✓ 已保存到: {episode_dir}")
        
        self.episodes.append({
            "id": episode_id,
            "path": str(episode_dir),
            "num_frames": len(frames),
        })
    
    def save_dataset_summary(self):
        """保存数据集摘要"""
        summary = {
            "num_episodes": len(self.episodes),
            "episodes": self.episodes,
            "robot_type": "bi_rm65_follower",
            "created_at": datetime.now().isoformat(),
        }
        
        with open(self.output_dir / "dataset_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ 数据集摘要已保存: {self.output_dir / 'dataset_summary.json'}")


def main():
    """主函数"""
    print("=" * 60)
    print("RM65 双臂数据录制工具")
    print("=" * 60)
    
    # 配置
    num_episodes = int(input("\n请输入要录制的片段数量 (建议 10-50): ") or "10")
    duration = int(input("每个片段的时长 (秒, 建议 20-40): ") or "20")
    fps = int(input("采样频率 (Hz, 建议 30): ") or "30")
    
    print(f"\n配置:")
    print(f"  片段数量: {num_episodes}")
    print(f"  每段时长: {duration}s")
    print(f"  采样频率: {fps} Hz")
    
    # 创建录制器
    recorder = RM65DataRecorder()
    
    try:
        # 连接机器人
        recorder.connect()
        
        # 录制片段
        for i in range(num_episodes):
            recorder.record_episode(i, duration=duration, fps=fps)
            
            if i < num_episodes - 1:
                print("\n准备录制下一个片段...")
                input("按回车继续,或 Ctrl+C 退出...")
        
        # 保存数据集摘要
        recorder.save_dataset_summary()
        
        print("\n" + "=" * 60)
        print(f"🎉 录制完成! 共 {num_episodes} 个片段")
        print(f"📁 保存位置: {recorder.output_dir}")
        print("=" * 60)
        print("\n下一步:")
        print("1. 查看录制的数据")
        print("2. 转换为 LeRobot 数据集格式")
        print("3. 训练策略模型")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  录制被中断")
    except Exception as e:
        print(f"\n\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        recorder.disconnect()


if __name__ == "__main__":
    main()
