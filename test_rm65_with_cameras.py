#!/usr/bin/env python
"""
RM65 双臂 + 相机配置示例

使用方法:
1. 运行 python -m lerobot.find_cameras 查看可用相机
2. 修改下面的相机索引
3. 运行此脚本测试
"""

from lerobot.robots.bi_rm65_follower import BiRM65FollowerConfig, BiRM65Follower
from lerobot.cameras.configs import OpenCVCameraConfig


def test_with_cameras():
    """测试 RM65 + 相机"""
    
    # 相机配置
    # 注意: 运行 python -m lerobot.find_cameras 查看你的相机索引
    cameras_config = {
        # 第三人称视角相机 (俯视或侧面拍摄整个工作区)
        "top": OpenCVCameraConfig(
            camera_index=0,  # 根据 find_cameras 结果修改
            fps=30,
            width=640,
            height=480,
        ),
        # 手腕相机 (安装在机械臂末端,第一人称视角)
        # "wrist": OpenCVCameraConfig(
        #     camera_index=1,  # 如果有第二个相机
        #     fps=30,
        #     width=640,
        #     height=480,
        # ),
    }
    
    # 机器人配置
    config = BiRM65FollowerConfig(
        id="rm65_with_cameras",
        left_arm_ip="169.254.128.20",
        right_arm_ip="169.254.128.21",
        port=8080,
        move_speed=30,
        cameras=cameras_config,  # 添加相机配置
    )
    
    print("=" * 60)
    print("RM65 相机测试")
    print("=" * 60)
    print(f"\n配置:")
    print(f"  左臂: {config.left_arm_ip}:{config.port}")
    print(f"  右臂: {config.right_arm_ip}:{config.port}")
    print(f"  相机: {list(cameras_config.keys())}")
    
    # 创建机器人
    robot = BiRM65Follower(config)
    
    try:
        print("\n正在连接...")
        robot.connect(calibrate=False)
        print("✓ 连接成功!")
        
        # 读取观察数据 (包含相机图像)
        print("\n正在读取观察数据 (关节 + 图像)...")
        obs = robot.get_observation()
        
        print(f"\n✓ 观察数据包含:")
        for key, value in obs.items():
            if key.endswith('.pos'):
                print(f"  {key}: {value:.2f}°")
            elif isinstance(value, type(obs.get('top'))):  # 图像数据
                import numpy as np
                if isinstance(value, np.ndarray):
                    print(f"  {key}: 图像 {value.shape} (高×宽×通道)")
        
        # 保存一张测试图像
        if 'top' in obs:
            import cv2
            import numpy as np
            img = obs['top']
            if isinstance(img, np.ndarray):
                cv2.imwrite('test_camera_frame.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                print(f"\n✓ 已保存测试图像: test_camera_frame.jpg")
        
        print("\n" + "=" * 60)
        print("相机配置成功!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if robot.is_connected:
            print("\n正在断开连接...")
            robot.disconnect()
            print("✓ 已断开")


if __name__ == "__main__":
    test_with_cameras()
