#!/usr/bin/env python
"""
RM65 双臂数据录制脚本 - 使用 LeRobot 官方 API

使用方法:
1. 运行: python record_rm65_demo.py --repo_id woosh/rm65_demo
2. 按提示手动移动机械臂
3. 按回车开始/停止录制
4. 数据自动保存为标准 LeRobot v3.0 格式

录制内容:
- 双臂关节角度 (12 个关节)
- 相机视频 (640×480@30fps, MP4编码)
- 时间戳同步
- Parquet格式数据
"""

import argparse
import time
from pathlib import Path

from lerobot.robots.bi_rm65_follower import BiRM65FollowerConfig, BiRM65Follower
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame


class RM65DataRecorder:
    """使用 LeRobot 官方 API 的 RM65 数据录制器"""
    
    def __init__(self, repo_id, root=None, fps=30):
        self.repo_id = repo_id
        self.fps = fps
        
        # 相机配置
        cameras_config = {
            "top": OpenCVCameraConfig(
                index_or_path=0,
                fps=fps,
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
            enable_right_gripper=True,  # 启用右臂夹爪
            gripper_device_id=9,
            gripper_force=60,
            gripper_speed=255,
            cameras=cameras_config,
        )
        
        self.robot = BiRM65Follower(self.config)
        self.dataset = None
        self.root = Path(root) if root else None
    
    def connect(self):
        """连接机器人"""
        print("\n正在连接 RM65 双臂机器人...")
        self.robot.connect(calibrate=False)
        print("✓ 连接成功!")
    
    def create_dataset(self, task_description="RM65 demonstration task"):
        """创建 LeRobot 数据集"""
        print(f"\n正在创建数据集: {self.repo_id}")
        
        # 从机器人获取 features 并转换为数据集格式
        from lerobot.datasets.pipeline_features import create_initial_features, aggregate_pipeline_dataset_features
        from lerobot.processor.pipeline import DataProcessorPipeline
        
        # 创建初始 features
        initial_features = create_initial_features(
            observation=self.robot.observation_features,
            action=self.robot.action_features,
        )
        
        # 使用空的pipeline转换features
        empty_pipeline = DataProcessorPipeline(steps=[])
        features = aggregate_pipeline_dataset_features(
            pipeline=empty_pipeline,
            initial_features=initial_features,
            use_videos=True,
        )
        
        # 创建数据集
        self.dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=self.fps,
            root=self.root,
            robot_type="bi_rm65_follower",
            features=features,
            use_videos=True,  # 使用视频编码
            image_writer_threads=4,  # 每个相机4个线程
        )
        
        # 启动图像写入器
        if hasattr(self.robot, "cameras") and len(self.robot.cameras) > 0:
            self.dataset.start_image_writer(
                num_processes=0,  # 使用线程而非进程
                num_threads=4 * len(self.robot.cameras),
            )
        
        print(f"✓ 数据集已创建: {self.dataset.root}")
    
    def record_episode(self, episode_index, duration=20):
        """
        录制一个演示片段
        
        Args:
            episode_index: 片段编号
            duration: 录制时长 (秒)
        """
        print(f"\n" + "=" * 60)
        print(f"录制片段 #{episode_index}")
        print("=" * 60)
        
        # 准备阶段
        print("\n请将机械臂移动到起始位置...")
        input("按回车键开始录制...")
        
        print(f"\n🔴 开始录制 ({duration} 秒, {self.fps} Hz)")
        print("请按住使能按钮并演示任务...")
        
        # 录制数据
        interval = 1.0 / self.fps
        num_frames = int(duration * self.fps)
        
        start_time = time.time()
        
        for frame_index in range(num_frames):
            frame_start = time.time()
            
            # 读取观察数据 (关节 + 图像)
            observation = self.robot.get_observation()
            
            # RM65 Follower模式: action = observation (没有独立控制)
            action = {k: v for k, v in observation.items() if not k.startswith("images.")}
            
            # 分别构建observation和action frame
            observation_frame = build_dataset_frame(self.dataset.features, observation, "observation")
            action_frame = build_dataset_frame(self.dataset.features, action, "action")
            frame = {**observation_frame, **action_frame, "task": "rm65_demo"}
            
            # 添加到数据集
            self.dataset.add_frame(frame)
            
            # 控制采样率
            elapsed = time.time() - frame_start
            if elapsed < interval:
                time.sleep(interval - elapsed)
            
            # 简单的进度显示
            if (frame_index + 1) % 30 == 0:  # 每秒显示一次
                print(f"  进度: {frame_index + 1}/{num_frames} 帧")
        
        actual_duration = time.time() - start_time
        actual_fps = num_frames / actual_duration
        
        print(f"\n✓ 录制完成!")
        print(f"  实际时长: {actual_duration:.2f}s")
        print(f"  实际帧率: {actual_fps:.1f} fps")
        print(f"  总帧数: {num_frames}")
        
        # 保存片段
        self.dataset.save_episode()
        
        return num_frames
    
    def disconnect(self):
        """断开机器人连接并关闭数据集"""
        print("\n正在关闭...")
        
        # 停止图像写入器
        if self.dataset is not None:
            self.dataset.stop_image_writer()
        
        # 断开机器人
        if self.robot is not None:
            self.robot.disconnect()
        
        print("✓ 已断开连接")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RM65 双臂数据录制工具")
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="数据集ID (例如: woosh/rm65_demo)"
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="数据集保存路径 (默认: ~/.cache/huggingface/lerobot/{repo_id})"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=10,
        help="录制的片段数量 (默认: 10)"
    )
    parser.add_argument(
        "--episode_duration",
        type=int,
        default=20,
        help="每个片段的时长(秒) (默认: 20)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="采样频率(Hz) (默认: 30)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="rm65_demo",
        help="任务描述 (默认: rm65_demo)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("RM65 双臂数据录制工具 (LeRobot v3.0 格式)")
    print("=" * 60)
    print(f"\n数据集: {args.repo_id}")
    print(f"片段数量: {args.num_episodes}")
    print(f"每段时长: {args.episode_duration}s")
    print(f"采样频率: {args.fps} Hz")
    
    # 创建录制器
    recorder = RM65DataRecorder(
        repo_id=args.repo_id,
        root=args.root,
        fps=args.fps,
    )
    
    try:
        # 连接机器人
        recorder.connect()
        
        # 创建数据集
        recorder.create_dataset(task_description=args.task)
        
        # 录制片段
        for i in range(args.num_episodes):
            recorder.record_episode(i, duration=args.episode_duration)
            
            if i < args.num_episodes - 1:
                print("\n准备录制下一个片段...")
                input("按回车继续,或 Ctrl+C 退出...")
        
        # LeRobot v3.0 不需要 consolidate(),数据已在 save_episode() 中保存
        print("\n✓ 数据集已保存")
        
        print("\n" + "=" * 60)
        print(f"🎉 录制完成! 共 {args.num_episodes} 个片段")
        print(f"📁 保存位置: {recorder.dataset.root}")
        print("=" * 60)
        print("\n数据集格式:")
        print("  ├── data/chunk-000/file-000.parquet")
        print("  ├── meta/info.json, stats.json")
        print("  └── videos/observation.images.top/chunk-000/file-000.mp4")
        print("\n下一步:")
        print(f"1. 上传到Hub: huggingface-cli upload {args.repo_id} {recorder.dataset.root}")
        print("2. 训练策略模型")
        
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
