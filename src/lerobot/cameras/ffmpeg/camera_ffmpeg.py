import logging
import subprocess
import time
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

import numpy as np

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..camera import Camera
from .configuration_ffmpeg import FFmpegCameraConfig

logger = logging.getLogger(__name__)


class FFmpegCamera(Camera):
    """
    Camera capture using FFmpeg for better performance.
    
    Uses subprocess to call ffmpeg and read raw frame data through pipe.
    Supports both synchronous and asynchronous frame reading.
    
    Example:
        ```python
        from lerobot.cameras.ffmpeg import FFmpegCamera, FFmpegCameraConfig
        
        config = FFmpegCameraConfig(
            index_or_path='/dev/video0',
            fps=30,
            width=1920,
            height=1080
        )
        camera = FFmpegCamera(config)
        camera.connect()
        
        frame = camera.read()
        print(frame.shape)  # (1080, 1920, 3)
        
        camera.disconnect()
        ```
    """
    
    def __init__(self, config: FFmpegCameraConfig):
        super().__init__(config)
        
        self.config = config
        self.index_or_path = config.index_or_path
        self.fps = config.fps
        self.pixel_format = config.pixel_format
        self.input_format = config.input_format
        
        self.process: subprocess.Popen | None = None
        self.frame_size: int = 0
        
        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: np.ndarray | None = None
        self.new_frame_event: Event = Event()
        
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.index_or_path})"
    
    @property
    def is_connected(self) -> bool:
        return self.process is not None and self.process.poll() is None
    
    def connect(self, warmup: bool = True):
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")
        
        # Build ffmpeg command
        cmd = [
            'ffmpeg',
            '-f', self.input_format,
            '-framerate', str(self.fps),
            '-video_size', f'{self.width}x{self.height}',
            '-i', str(self.index_or_path),
            '-pix_fmt', self.pixel_format,
            '-f', 'rawvideo',
            '-'
        ]
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=10**8
            )
        except Exception as e:
            raise ConnectionError(f"Failed to start ffmpeg for {self}: {e}")
        
        # Calculate frame size
        if self.pixel_format == 'rgb24':
            self.frame_size = self.width * self.height * 3
        else:
            raise ValueError(f"Unsupported pixel format: {self.pixel_format}")
        
        if warmup:
            start_time = time.time()
            while time.time() - start_time < 1.0:
                try:
                    self.read()
                    time.sleep(0.1)
                except Exception:
                    pass
        
        logger.info(f"{self} connected via ffmpeg.")
    
    def read(self) -> np.ndarray:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        start_time = time.perf_counter()
        
        try:
            raw_frame = self.process.stdout.read(self.frame_size)
        except Exception as e:
            raise RuntimeError(f"{self} read failed: {e}")
        
        if len(raw_frame) != self.frame_size:
            raise RuntimeError(
                f"{self} incomplete frame: got {len(raw_frame)} bytes, expected {self.frame_size}"
            )
        
        # Convert raw bytes to numpy array
        frame = np.frombuffer(raw_frame, dtype=np.uint8)
        frame = frame.reshape((self.height, self.width, 3))
        
        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} read took: {read_duration_ms:.1f}ms")
        
        return frame
    
    def _read_loop(self):
        while not self.stop_event.is_set():
            try:
                frame = self.read()
                
                with self.frame_lock:
                    self.latest_frame = frame
                self.new_frame_event.set()
                
            except DeviceNotConnectedError:
                break
            except Exception as e:
                logger.warning(f"Error reading frame in background thread for {self}: {e}")
    
    def _start_read_thread(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.1)
        if self.stop_event is not None:
            self.stop_event.set()
        
        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, args=(), name=f"{self}_read_loop")
        self.thread.daemon = True
        self.thread.start()
    
    def _stop_read_thread(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        
        self.thread = None
        self.stop_event = None
    
    def async_read(self, timeout_ms: float = 200) -> np.ndarray:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()
        
        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            thread_alive = self.thread is not None and self.thread.is_alive()
            raise TimeoutError(
                f"Timed out waiting for frame from camera {self} after {timeout_ms} ms. "
                f"Read thread alive: {thread_alive}."
            )
        
        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()
        
        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no frame available for {self}.")
        
        return frame
    
    def disconnect(self):
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(f"{self} not connected.")
        
        if self.thread is not None:
            self._stop_read_thread()
        
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
        
        logger.info(f"{self} disconnected.")
    
    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """Find available video devices for ffmpeg."""
        found_cameras_info = []
        
        # Scan /dev/video* on Linux
        possible_paths = sorted(Path("/dev").glob("video*"), key=lambda p: p.name)
        
        for path in possible_paths:
            camera_info = {
                "name": f"FFmpeg Camera @ {path}",
                "type": "FFmpeg",
                "id": str(path),
                "backend_api": "ffmpeg",
                "default_stream_profile": {
                    "format": "rgb24",
                    "width": 1920,
                    "height": 1080,
                    "fps": 30,
                },
            }
            found_cameras_info.append(camera_info)
        
        return found_cameras_info
