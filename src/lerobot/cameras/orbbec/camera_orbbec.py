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
Provides the OrbbecCamera class for capturing frames from Orbbec depth cameras (e.g., 大白Pro).
"""

import logging
import time
from threading import Event, Lock, Thread
from typing import Any

import cv2
import numpy as np

try:
    from pyorbbecsdk import Context, Pipeline, Config, OBSensorType, OBFormat
except ImportError as e:
    logging.warning(f"Could not import pyorbbecsdk: {e}. Orbbec cameras will not be available.")
    Context = Pipeline = Config = None

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..camera import Camera
from ..configs import ColorMode
from ..utils import get_cv2_rotation
from .configuration_orbbec import OrbbecCameraConfig

logger = logging.getLogger(__name__)


class OrbbecCamera(Camera):
    """
    Manages interactions with Orbbec depth cameras (e.g., 大白Pro).
    
    This class provides an interface similar to RealSenseCamera but for Orbbec devices,
    using the pyorbbecsdk library. It supports capturing color frames and optionally
    depth maps.
    
    Example:
        ```python
        from lerobot.cameras.orbbec import OrbbecCamera, OrbbecCameraConfig
        
        # Basic usage
        config = OrbbecCameraConfig(device_index=0)
        camera = OrbbecCamera(config)
        camera.connect()
        
        # Read frame
        color_image = camera.read()
        print(color_image.shape)
        
        # Async read
        async_image = camera.async_read()
        
        camera.disconnect()
        ```
    """
    
    def __init__(self, config: OrbbecCameraConfig):
        super().__init__(config)
        
        self.config = config
        self.device_index = config.device_index
        self.fps = config.fps
        self.color_mode = config.color_mode
        self.use_depth = config.use_depth
        
        self.ob_ctx: Context | None = None
        self.ob_pipeline: Pipeline | None = None
        self.ob_config: Config | None = None
        
        # Async reading
        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: np.ndarray | None = None
        self.new_frame_event: Event = Event()
        
        self.rotation: int | None = get_cv2_rotation(config.rotation)
        
        if self.height and self.width:
            self.capture_width, self.capture_height = self.width, self.height
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.capture_width, self.capture_height = self.height, self.width
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(device_{self.device_index})"
    
    @property
    def is_connected(self) -> bool:
        """Checks if the camera pipeline is started."""
        return self.ob_pipeline is not None
    
    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """
        Detects available Orbbec cameras connected to the system.
        
        Returns:
            List of dictionaries containing camera information
        """
        if Context is None:
            logger.warning("pyorbbecsdk not installed. Cannot find cameras.")
            return []
        
        try:
            ctx = Context()
            device_list = ctx.query_devices()
            
            cameras = []
            for i in range(device_list.get_count()):
                device = device_list.get_device_by_index(i)
                device_info = device.get_device_info()
                
                cameras.append({
                    "index": i,
                    "name": device_info.get_name(),
                    "serial_number": device_info.get_serial_number(),
                    "vendor_id": device_info.get_vid(),
                    "product_id": device_info.get_pid(),
                })
            
            return cameras
            
        except Exception as e:
            logger.error(f"Error finding Orbbec cameras: {e}")
            return []
    
    def connect(self, warmup: bool = True):
        """
        Connects to the Orbbec camera.
        
        Raises:
            DeviceAlreadyConnectedError: If already connected
            ConnectionError: If camera not found or failed to start
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")
        
        if Context is None:
            raise ImportError(
                "pyorbbecsdk is not installed. Install it with: pip install pyorbbecsdk"
            )
        
        try:
            # Initialize Orbbec context and pipeline
            self.ob_ctx = Context()
            device_list = self.ob_ctx.query_devices()
            
            if device_list.get_count() == 0:
                raise ConnectionError("No Orbbec devices found!")
            
            if self.device_index >= device_list.get_count():
                raise ConnectionError(
                    f"Device index {self.device_index} out of range. "
                    f"Found {device_list.get_count()} devices."
                )
            
            device = device_list.get_device_by_index(self.device_index)
            self.ob_pipeline = Pipeline(device)
            
            # Configure streams
            self.ob_config = Config()
            
            # Get available color profiles
            color_profiles = self.ob_pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            if color_profiles.get_count() == 0:
                raise ConnectionError(f"No color profiles available for {self}")
            
            # Find matching profile or use first one
            color_profile = None
            for i in range(color_profiles.get_count()):
                profile = color_profiles.get_profile(i)
                video_profile = profile.as_video_stream_profile()
                if (video_profile.get_width() == self.width and 
                    video_profile.get_height() == self.height and
                    video_profile.get_fps() == self.fps):
                    color_profile = profile
                    break
            
            if color_profile is None:
                logger.warning(
                    f"No exact match for {self.width}x{self.height}@{self.fps}. "
                    f"Using first available profile."
                )
                color_profile = color_profiles.get_profile(0)
                video_profile = color_profile.as_video_stream_profile()
                self.width = video_profile.get_width()
                self.height = video_profile.get_height()
                self.fps = video_profile.get_fps()
            
            self.ob_config.enable_stream(color_profile)
            
            # TODO: Add depth stream support
            # if self.use_depth:
            #     depth_profiles = self.ob_pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            #     ...
            
            # Start pipeline
            self.ob_pipeline.start(self.ob_config)
            
            logger.info(f"{self} connected at {self.width}x{self.height}@{self.fps}fps")
            
            if warmup:
                time.sleep(0.5)
                for _ in range(10):
                    self.read()
                    time.sleep(0.1)
            
        except Exception as e:
            self.ob_pipeline = None
            self.ob_ctx = None
            raise ConnectionError(f"Failed to connect to {self}: {e}") from e
    
    def read(self, temporary_color_mode: ColorMode | None = None) -> np.ndarray:
        """
        Captures and returns a single frame from the camera.
        
        Args:
            temporary_color_mode: Override color mode for this read
            
        Returns:
            Frame as numpy array
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected. Call `connect()` first.")
        
        try:
            # Wait for frames (timeout: 1000ms)
            frameset = self.ob_pipeline.wait_for_frames(1000)
            if frameset is None:
                raise RuntimeError(f"Failed to get frames from {self}")
            
            # Get color frame
            color_frame = frameset.get_color_frame()
            if color_frame is None:
                raise RuntimeError(f"Failed to get color frame from {self}")
            
            # Convert to numpy array
            width = color_frame.get_width()
            height = color_frame.get_height()
            data = np.asanyarray(color_frame.get_data())
            
            # Reshape based on format
            # Orbbec typically outputs RGB888 or YUYV
            format_type = color_frame.get_format()
            if format_type == OBFormat.RGB:
                frame = data.reshape((height, width, 3))
            elif format_type == OBFormat.YUYV:
                # Convert YUYV to RGB
                frame = cv2.cvtColor(data.reshape((height, width, 2)), cv2.COLOR_YUV2RGB_YUYV)
            else:
                logger.warning(f"Unsupported format {format_type}, attempting reshape...")
                frame = data.reshape((height, width, 3))
            
            # Handle color mode conversion
            color_mode = temporary_color_mode or self.color_mode
            if color_mode == ColorMode.BGR:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Apply rotation if needed
            if self.rotation is not None:
                frame = cv2.rotate(frame, self.rotation)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error reading from {self}: {e}")
            raise
    
    def read_depth(self) -> np.ndarray | None:
        """
        Captures and returns a depth map.
        
        TODO: Implement depth capture
        
        Returns:
            Depth map as uint16 numpy array (millimeters), or None if not available
        """
        if not self.use_depth:
            logger.warning(f"Depth not enabled for {self}. Set use_depth=True in config.")
            return None
        
        # TODO: Implement depth reading
        logger.warning("Depth capture not yet implemented for OrbbecCamera")
        return None
    
    def async_read(self) -> np.ndarray:
        """
        Returns the latest frame from the background thread.
        
        If no frame is available yet, blocks until one arrives.
        """
        if self.thread is None or not self.thread.is_alive():
            self._start_async_thread()
        
        self.new_frame_event.wait()
        with self.frame_lock:
            return self.latest_frame.copy()
    
    def _start_async_thread(self):
        """Starts the background frame reading thread."""
        if self.thread is not None and self.thread.is_alive():
            return
        
        self.stop_event = Event()
        self.thread = Thread(target=self._async_loop, daemon=True)
        self.thread.start()
        logger.debug(f"Started async thread for {self}")
    
    def _async_loop(self):
        """Background loop for continuous frame reading."""
        while not self.stop_event.is_set():
            try:
                frame = self.read()
                with self.frame_lock:
                    self.latest_frame = frame
                self.new_frame_event.set()
            except Exception as e:
                logger.error(f"Error in async loop for {self}: {e}")
                time.sleep(0.1)
    
    def disconnect(self):
        """Stops the camera and cleans up resources."""
        if not self.is_connected:
            return
        
        # Stop async thread
        if self.thread is not None:
            self.stop_event.set()
            self.thread.join(timeout=2.0)
            self.thread = None
        
        # Stop pipeline
        if self.ob_pipeline is not None:
            try:
                self.ob_pipeline.stop()
            except Exception as e:
                logger.warning(f"Error stopping pipeline: {e}")
            self.ob_pipeline = None
        
        self.ob_ctx = None
        self.ob_config = None
        
        logger.info(f"{self} disconnected")
