#!/usr/bin/env python
"""
测试 LeRobot 项目是否成功使用 FFmpeg 作为视频采集后端
"""

import time
from lerobot.robots.bi_rm65_follower import BiRM65FollowerConfig, BiRM65Follower
from lerobot.cameras.ffmpeg import FFmpegCameraConfig


def test_lerobot_ffmpeg_integration():
    """测试 LeRobot 与 FFmpeg 相机集成"""
    
    print("=" * 60)
    print("LeRobot FFmpeg 集成测试")
    print("=" * 60)
    
    # 创建 FFmpeg 相机配置
    cameras_config = {
        "top": FFmpegCameraConfig(
            index_or_path="/dev/video0",
            fps=30,
            width=1920,
            height=1080,
        ),
        "wrist": FFmpegCameraConfig(
            index_or_path="/dev/video2",
            fps=30,
            width=1920,
            height=1080,
        ),
    }
    
    # 创建机器人配置
    robot_config = BiRM65FollowerConfig(
        id="test_ffmpeg",
        left_arm_ip="169.254.128.20",
        right_arm_ip="169.254.128.21",
        port=8080,
        cameras=cameras_config,
    )
    
    robot = BiRM65Follower(robot_config)
    
    try:
        print("\n1. 连接机器人和相机...")
        robot.connect(calibrate=False)
        print("   ✓ 连接成功")
        
        print("\n2. 检查相机类型...")
        for cam_name, cam in robot.cameras.items():
            print(f"   {cam_name}: {type(cam).__name__}")
            assert "FFmpeg" in type(cam).__name__, f"相机 {cam_name} 不是 FFmpeg 类型！"
        print("   ✓ 所有相机均为 FFmpeg 类型")
        
        print("\n3. 测试图像采集 (30帧)...")
        start_time = time.time()
        frame_count = 30
        
        for i in range(frame_count):
            obs = robot.get_observation()
            
            # 检查图像分辨率
            if i == 0:
                for cam_name in ["top", "wrist"]:
                    img = obs[cam_name]
                    h, w, c = img.shape
                    print(f"   {cam_name}: {w}x{h}x{c}")
                    assert (h, w) == (1080, 1920), f"分辨率错误: {w}x{h}"
        
        elapsed = time.time() - start_time
        actual_fps = frame_count / elapsed
        
        print(f"\n   采集 {frame_count} 帧")
        print(f"   耗时: {elapsed:.2f}s")
        print(f"   实际帧率: {actual_fps:.2f} fps")
        print("   ✓ 图像采集成功")
        
        print("\n" + "=" * 60)
        print("✓ 测试通过！LeRobot 已成功使用 FFmpeg 作为视频采集后端")
        print("=" * 60)
        print("\n配置摘要:")
        print(f"  - 相机类型: FFmpeg (v4l2 + mjpeg)")
        print(f"  - 分辨率: 1920x1080")
        print(f"  - 帧率: ~{actual_fps:.1f} fps")
        print(f"  - 设备: /dev/video0, /dev/video2")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if robot.is_connected:
            robot.disconnect()
            print("\n✓ 已断开连接")
    
    return True


if __name__ == "__main__":
    success = test_lerobot_ffmpeg_integration()
    exit(0 if success else 1)
