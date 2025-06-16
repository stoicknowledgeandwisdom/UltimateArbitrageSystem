"""Operations module for ML optimization infrastructure."""

from .device_manager import DeviceManager
from .ray_serve_manager import RayServeManager
from .shadow_models import ShadowModelManager
from .drift_detector import DriftDetector

__all__ = ['DeviceManager', 'RayServeManager', 'ShadowModelManager', 'DriftDetector']

