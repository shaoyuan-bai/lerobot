from dataclasses import dataclass, field

from ..configs import CameraConfig


@dataclass
class FFmpegCameraConfig(CameraConfig):
    """Configuration for FFmpeg-based camera capture.
    
    Args:
        index_or_path: Camera device path (e.g., '/dev/video0' on Linux).
        fps: Frames per second for capture (default: 30).
        width: Frame width in pixels (default: 640).
        height: Frame height in pixels (default: 480).
        pixel_format: FFmpeg pixel format (default: 'rgb24').
        input_format: Input format for ffmpeg (default: 'v4l2' on Linux).
    """
    index_or_path: str | int = 0
    fps: int | None = 30
    width: int | None = 640
    height: int | None = 480
    pixel_format: str = "rgb24"
    input_format: str = "v4l2"
