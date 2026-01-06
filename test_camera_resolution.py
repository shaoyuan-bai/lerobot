#!/usr/bin/env python
"""
测试摄像头分辨率和格式

用法:
    python test_camera_resolution.py --camera-index 0
    python test_camera_resolution.py --camera-index 2 --test-resolution 1920 1080
"""

import argparse
import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def list_camera_formats(camera_index: int):
    """列出摄像头支持的所有分辨率"""
    logger.info(f"\n{'='*60}")
    logger.info(f"检测摄像头 {camera_index} 的支持格式")
    logger.info(f"{'='*60}\n")
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error(f"❌ 无法打开摄像头 {camera_index}")
        return
    
    # 常见分辨率列表（从高到低）
    common_resolutions = [
        (3840, 2160, "4K UHD"),
        (2560, 1440, "2K QHD"),
        (1920, 1080, "Full HD 1080p"),
        (1280, 720, "HD 720p"),
        (640, 480, "VGA"),
        (640, 360, "360p"),
    ]
    
    supported_resolutions = []
    
    logger.info("🔍 测试常见分辨率...")
    for width, height, name in common_resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if actual_width == width and actual_height == height:
            # 尝试读取一帧来验证
            ret, frame = cap.read()
            if ret and frame is not None:
                fps = cap.get(cv2.CAP_PROP_FPS)
                supported_resolutions.append((width, height, name, fps))
                logger.info(f"  ✅ {name:20s} {width}x{height} @ {fps:.1f} fps")
            else:
                logger.info(f"  ⚠️  {name:20s} {width}x{height} (设置成功但无法读取)")
        else:
            logger.info(f"  ❌ {name:20s} {width}x{height} (不支持)")
    
    cap.release()
    
    if supported_resolutions:
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 支持的分辨率总结")
        logger.info(f"{'='*60}")
        max_width, max_height, max_name, max_fps = supported_resolutions[0]
        logger.info(f"🏆 最高分辨率: {max_name} ({max_width}x{max_height}) @ {max_fps:.1f} fps")
        logger.info(f"📝 共支持 {len(supported_resolutions)} 种分辨率\n")
    else:
        logger.warning("⚠️  未检测到任何支持的分辨率")


def test_resolution(camera_index: int, width: int, height: int, duration: int = 5):
    """测试指定分辨率并显示实时画面"""
    logger.info(f"\n{'='*60}")
    logger.info(f"测试摄像头 {camera_index} 在 {width}x{height} 分辨率下的表现")
    logger.info(f"{'='*60}\n")
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error(f"❌ 无法打开摄像头 {camera_index}")
        return
    
    # 设置分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    logger.info(f"📹 请求分辨率: {width}x{height}")
    logger.info(f"📹 实际分辨率: {actual_width}x{actual_height}")
    logger.info(f"📹 帧率: {fps:.1f} fps")
    
    if actual_width != width or actual_height != height:
        logger.warning(f"⚠️  实际分辨率与请求不符！")
        user_input = input(f"是否继续测试 {actual_width}x{actual_height}? (y/n): ")
        if user_input.lower() != 'y':
            cap.release()
            return
    
    logger.info(f"\n🎥 开始录制 {duration} 秒...")
    logger.info("💡 提示: 将显示实时画面，按 'q' 键提前退出\n")
    
    frame_count = 0
    import time
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("❌ 读取帧失败")
            break
        
        frame_count += 1
        elapsed = time.time() - start_time
        
        # 添加信息到画面
        cv2.putText(frame, f"Resolution: {actual_width}x{actual_height}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {frame_count / elapsed:.1f}", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Time: {elapsed:.1f}s / {duration}s", 
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示画面
        cv2.imshow(f"Camera {camera_index} Test", frame)
        
        # 检查是否按下 'q' 键或时间到
        if cv2.waitKey(1) & 0xFF == ord('q') or elapsed >= duration:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    actual_fps = frame_count / elapsed
    logger.info(f"\n✅ 测试完成!")
    logger.info(f"📊 统计信息:")
    logger.info(f"  - 总帧数: {frame_count}")
    logger.info(f"  - 实际帧率: {actual_fps:.2f} fps")
    logger.info(f"  - 分辨率: {actual_width}x{actual_height}")
    logger.info(f"  - 每帧大小: ~{(actual_width * actual_height * 3) / 1024 / 1024:.2f} MB (未压缩)")


def main():
    parser = argparse.ArgumentParser(description="测试摄像头分辨率和格式")
    parser.add_argument("--camera-index", type=int, default=0, help="摄像头索引 (默认: 0)")
    parser.add_argument("--test-resolution", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"),
                        help="测试指定分辨率 (例如: --test-resolution 1920 1080)")
    parser.add_argument("--duration", type=int, default=5, help="测试时长（秒，默认: 5）")
    args = parser.parse_args()
    
    if args.test_resolution:
        width, height = args.test_resolution
        test_resolution(args.camera_index, width, height, args.duration)
    else:
        list_camera_formats(args.camera_index)
        
        # 询问是否测试最高分辨率
        user_input = input("\n是否测试最高分辨率？(y/n): ")
        if user_input.lower() == 'y':
            # 重新检测最高分辨率
            cap = cv2.VideoCapture(args.camera_index)
            common_resolutions = [
                (3840, 2160), (2560, 1440), (1920, 1080), 
                (1280, 720), (640, 480), (640, 360)
            ]
            for width, height in common_resolutions:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if actual_width == width and actual_height == height:
                    ret, frame = cap.read()
                    if ret:
                        cap.release()
                        test_resolution(args.camera_index, width, height, args.duration)
                        break
            else:
                cap.release()
                logger.warning("⚠️  未找到支持的最高分辨率")


if __name__ == "__main__":
    main()
