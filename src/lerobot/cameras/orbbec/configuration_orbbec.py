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
Configuration for Orbbec (大白Pro) depth cameras.
"""

from dataclasses import dataclass, field

from ..camera import CameraConfig
from ..configs import ColorMode, Cv2Rotation


@dataclass
class OrbbecCameraConfig(CameraConfig):
    """
    Configuration for Orbbec cameras (e.g., 大白Pro).
    
    Attributes:
        device_index: Device index (0 for first camera, 1 for second, etc.)
        use_depth: Whether to capture depth data in addition to color
        depth_unit: Depth unit in millimeters (default: 1.0 for mm)
    """
    
    # Orbbec specific settings
    device_index: int = 0
    use_depth: bool = False  # TODO: Enable depth capture
    depth_unit: float = 1.0  # Depth in millimeters
    
    # Override defaults from CameraConfig
    fps: int = 30
    width: int = 640
    height: int = 480
    color_mode: ColorMode = ColorMode.RGB
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    
    def __post_init__(self):
        super().__post_init__()
