#!/usr/bin/env python
"""Test FFmpeg camera capture performance."""

import time
from lerobot.cameras.ffmpeg import FFmpegCamera, FFmpegCameraConfig

def test_ffmpeg_camera(device_path: str = "/dev/video0", width: int = 1920, height: int = 1080, fps: int = 30):
    """Test FFmpeg camera capture."""
    print(f"Testing FFmpeg camera: {device_path}")
    print(f"Resolution: {width}x{height} @ {fps}fps")
    
    config = FFmpegCameraConfig(
        index_or_path=device_path,
        fps=fps,
        width=width,
        height=height,
        pixel_format="rgb24",
        input_format="v4l2"
    )
    
    camera = FFmpegCamera(config)
    
    try:
        print("\nConnecting...")
        camera.connect()
        print("✓ Connected")
        
        print(f"\nReading {fps * 5} frames (5 seconds)...")
        start_time = time.time()
        frame_count = 0
        target_frames = fps * 5
        
        while frame_count < target_frames:
            frame = camera.read()
            frame_count += 1
            
            if frame_count % fps == 0:
                elapsed = time.time() - start_time
                actual_fps = frame_count / elapsed
                print(f"  {frame_count}/{target_frames} frames - {actual_fps:.2f} fps")
        
        elapsed = time.time() - start_time
        actual_fps = frame_count / elapsed
        
        print(f"\n✓ Test complete")
        print(f"  Total frames: {frame_count}")
        print(f"  Duration: {elapsed:.2f}s")
        print(f"  Average FPS: {actual_fps:.2f}")
        print(f"  Frame shape: {frame.shape}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if camera.is_connected:
            camera.disconnect()
            print("\n✓ Disconnected")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test FFmpeg camera")
    parser.add_argument("--device", default="/dev/video0", help="Camera device path")
    parser.add_argument("--width", type=int, default=1920, help="Frame width")
    parser.add_argument("--height", type=int, default=1080, help="Frame height")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    
    args = parser.parse_args()
    
    test_ffmpeg_camera(args.device, args.width, args.height, args.fps)
