# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from .camera import Camera
from .configs import CameraConfig, ColorMode, Cv2Rotation
from .ffmpeg import FFmpegCamera, FFmpegCameraConfig
from .utils import make_cameras_from_configs

# Register camera types to draccus choice system
from .opencv.configuration_opencv import OpenCVCameraConfig

CameraConfig.register_subclass("opencv", OpenCVCameraConfig)
CameraConfig.register_subclass("ffmpeg", FFmpegCameraConfig)
