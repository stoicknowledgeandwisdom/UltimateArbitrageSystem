"""Device Manager for GPU/TPU auto-detection and optimization."""

import torch
import logging
import psutil
import subprocess
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import threading
import time
from datetime import datetime
import json


class DeviceType(Enum):
    """Supported device types."""
    CPU = "cpu"
    CUDA = "cuda"
    TPU = "tpu"
    MPS = "mps"  # Apple Metal Performance Shaders
    XLA = "xla"  # XLA accelerated devices


@dataclass
class DeviceInfo:
    """Information about a compute device."""
    device_type: DeviceType
    device_id: int
    name: str
    memory_total: int  # MB
    memory_available: int  # MB
    compute_capability: Optional[str] = None
    utilization: float = 0.0  # 0-100%
    temperature: Optional[float] = None  # Celsius
    power_usage: Optional[float] = None  # Watts
    is_available: bool = True
    tensor_cores: bool = False


@dataclass
class DeviceManagerConfig:
    """Configuration for device manager."""
    auto_detect: bool = True
    preferred_device_type: Optional[DeviceType] = None
    memory_fraction: float = 0.8  # Fraction of GPU memory to use
    enable_mixed_precision: bool = True
    monitor_interval: float = 30.0  # seconds
    temperature_threshold: float = 85.0  # Celsius
    memory_threshold: float = 0.9  # 90% memory usage threshold
    enable_device_monitoring: bool = True


class DeviceManager:
    """Manages compute devices for ML optimization."""
    
    def __init__(self, config: DeviceManagerConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Device information
        self.devices: Dict[str, DeviceInfo] = {}
        self.primary_device: Optional[str] = None
        self.device_pool: List[str] = []
        
        # Monitoring
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.device_stats_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Performance optimization settings
        self.optimization_settings: Dict[str, Any] = {}
        
        # Initialize device detection
        if config.auto_detect:
            self._detect_devices()
            self._configure_optimization()
            
            if config.enable_device_monitoring:
                self._start_monitoring()
    
    def _detect_devices(self):
        """Detect all available compute devices."""
        self.logger.info("Detecting available compute devices...")
        
        # Detect CPU
        self._detect_cpu()
        
        # Detect CUDA devices
        if torch.cuda.is_available():
            self._detect_cuda_devices()
        
        # Detect MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self._detect_mps_device()
        
        # Detect TPU (if available)
        try:
            self._detect_tpu_devices()
        except ImportError:
            self.logger.debug("TPU support not available")
        
        # Select primary device
        self._select_primary_device()
        
        self.logger.info(f"Detected {len(self.devices)} devices, primary: {self.primary_device}")
    
    def _detect_cpu(self):
        """Detect CPU information."""
        try:
            cpu_info = DeviceInfo(
                device_type=DeviceType.CPU,
                device_id=0,
                name=f"CPU ({psutil.cpu_count()} cores)",
                memory_total=int(psutil.virtual_memory().total / 1024 / 1024),
                memory_available=int(psutil.virtual_memory().available / 1024 / 1024),
                utilization=psutil.cpu_percent()
            )
            self.devices["cpu:0"] = cpu_info
            self.device_pool.append("cpu:0")
            
        except Exception as e:
            self.logger.error(f"Failed to detect CPU: {e}")
    
    def _detect_cuda_devices(self):
        """Detect CUDA GPU devices."""
        try:
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                
                # Get memory info
                torch.cuda.set_device(i)
                memory_total = torch.cuda.get_device_properties(i).total_memory // 1024 // 1024
                memory_free = torch.cuda.memory_reserved(i) // 1024 // 1024
                memory_available = memory_total - memory_free
                
                # Check for tensor cores
                tensor_cores = props.major >= 7  # Volta and newer
                
                cuda_info = DeviceInfo(
                    device_type=DeviceType.CUDA,
                    device_id=i,
                    name=props.name,
                    memory_total=memory_total,
                    memory_available=memory_available,
                    compute_capability=f"{props.major}.{props.minor}",
                    tensor_cores=tensor_cores
                )
                
                device_key = f"cuda:{i}"
                self.devices[device_key] = cuda_info
                self.device_pool.append(device_key)
                
                self.logger.info(f"Detected CUDA device {i}: {props.name} ({memory_total}MB)")
                
        except Exception as e:
            self.logger.error(f"Failed to detect CUDA devices: {e}")
    
    def _detect_mps_device(self):
        """Detect Apple MPS device."""
        try:
            # MPS doesn't provide detailed memory info, use system memory as estimate
            system_memory = int(psutil.virtual_memory().total / 1024 / 1024)
            
            mps_info = DeviceInfo(
                device_type=DeviceType.MPS,
                device_id=0,
                name="Apple Metal Performance Shaders",
                memory_total=system_memory // 2,  # Rough estimate
                memory_available=system_memory // 2
            )
            
            self.devices["mps:0"] = mps_info
            self.device_pool.append("mps:0")
            
            self.logger.info("Detected MPS device (Apple Silicon)")
            
        except Exception as e:
            self.logger.error(f"Failed to detect MPS device: {e}")
    
    def _detect_tpu_devices(self):
        """Detect TPU devices (Google Cloud TPU)."""
        try:
            import torch_xla.core.xla_model as xm
            
            # Check if TPU is available
            if xm.xrt_world_size() > 0:
                for i in range(xm.xrt_world_size()):
                    tpu_info = DeviceInfo(
                        device_type=DeviceType.TPU,
                        device_id=i,
                        name=f"TPU Core {i}",
                        memory_total=8192,  # TPU v3/v4 typically have 8GB HBM per core
                        memory_available=8192
                    )
                    
                    device_key = f"tpu:{i}"
                    self.devices[device_key] = tpu_info
                    self.device_pool.append(device_key)
                    
                    self.logger.info(f"Detected TPU core {i}")
                    
        except ImportError:
            self.logger.debug("TPU libraries not available")
        except Exception as e:
            self.logger.error(f"Failed to detect TPU devices: {e}")
    
    def _select_primary_device(self):
        """Select the primary device for computation."""
        if self.config.preferred_device_type:
            # Look for preferred device type
            for device_key, device_info in self.devices.items():
                if device_info.device_type == self.config.preferred_device_type:
                    self.primary_device = device_key
                    return
        
        # Auto-select based on capability
        # Priority: TPU > CUDA (with tensor cores) > CUDA > MPS > CPU
        device_priority = {
            DeviceType.TPU: 5,
            DeviceType.CUDA: 4,
            DeviceType.MPS: 3,
            DeviceType.CPU: 1
        }
        
        best_device = None
        best_score = -1
        
        for device_key, device_info in self.devices.items():
            score = device_priority.get(device_info.device_type, 0)
            
            # Bonus for tensor cores
            if device_info.tensor_cores:
                score += 1
            
            # Bonus for more memory
            if device_info.memory_total > 8000:  # > 8GB
                score += 0.5
            
            if score > best_score:
                best_score = score
                best_device = device_key
        
        self.primary_device = best_device
    
    def _configure_optimization(self):
        """Configure optimization settings based on detected devices."""
        if not self.primary_device:
            return
        
        primary_info = self.devices[self.primary_device]
        
        # Configure PyTorch optimizations
        if primary_info.device_type == DeviceType.CUDA:
            # Enable CUDA optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Configure memory fraction
            device_id = primary_info.device_id
            memory_fraction = int(primary_info.memory_total * self.config.memory_fraction)
            torch.cuda.set_per_process_memory_fraction(self.config.memory_fraction, device_id)
            
            # Enable mixed precision if supported
            if self.config.enable_mixed_precision and primary_info.tensor_cores:
                self.optimization_settings['mixed_precision'] = True
                self.logger.info("Enabled mixed precision training (tensor cores detected)")
            
        elif primary_info.device_type == DeviceType.MPS:
            # MPS optimizations
            torch.backends.mps.allow_fallback = True
            
        # Set number of threads for CPU
        if primary_info.device_type == DeviceType.CPU:
            torch.set_num_threads(psutil.cpu_count())
        
        self.logger.info(f"Configured optimizations for {self.primary_device}")
    
    def _start_monitoring(self):
        """Start device monitoring thread."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Started device monitoring")
    
    def _monitoring_loop(self):
        """Device monitoring loop."""
        while self.monitoring_active:
            try:
                self._update_device_stats()
                time.sleep(self.config.monitor_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5.0)  # Reduced interval on error
    
    def _update_device_stats(self):
        """Update device statistics."""
        current_time = datetime.now()
        
        for device_key, device_info in self.devices.items():
            try:
                stats = {
                    'timestamp': current_time.isoformat(),
                    'utilization': 0.0,
                    'memory_used': 0,
                    'memory_available': device_info.memory_available,
                    'temperature': None,
                    'power_usage': None
                }
                
                if device_info.device_type == DeviceType.CUDA:
                    # Update CUDA stats
                    device_id = device_info.device_id
                    
                    # Memory usage
                    memory_used = torch.cuda.memory_allocated(device_id) // 1024 // 1024
                    memory_cached = torch.cuda.memory_reserved(device_id) // 1024 // 1024
                    
                    stats['memory_used'] = memory_used
                    stats['memory_cached'] = memory_cached
                    stats['memory_available'] = device_info.memory_total - memory_cached
                    
                    # Try to get GPU utilization using nvidia-ml-py or nvidia-smi
                    try:
                        utilization = self._get_gpu_utilization(device_id)
                        stats['utilization'] = utilization
                        device_info.utilization = utilization
                    except:
                        pass
                    
                elif device_info.device_type == DeviceType.CPU:
                    # Update CPU stats
                    stats['utilization'] = psutil.cpu_percent()
                    memory = psutil.virtual_memory()
                    stats['memory_used'] = int((memory.total - memory.available) / 1024 / 1024)
                    stats['memory_available'] = int(memory.available / 1024 / 1024)
                    
                    device_info.utilization = stats['utilization']
                    device_info.memory_available = stats['memory_available']
                
                # Store stats history
                if device_key not in self.device_stats_history:
                    self.device_stats_history[device_key] = []
                
                self.device_stats_history[device_key].append(stats)
                
                # Keep only last 1000 entries
                if len(self.device_stats_history[device_key]) > 1000:
                    self.device_stats_history[device_key] = self.device_stats_history[device_key][-1000:]
                
                # Check thresholds
                self._check_device_thresholds(device_key, stats)
                
            except Exception as e:
                self.logger.error(f"Error updating stats for {device_key}: {e}")
    
    def _get_gpu_utilization(self, device_id: int) -> float:
        """Get GPU utilization using nvidia-smi."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits', f'--id={device_id}'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return float(result.stdout.strip())
        except:
            pass
        return 0.0
    
    def _check_device_thresholds(self, device_key: str, stats: Dict[str, Any]):
        """Check device thresholds and log warnings."""
        device_info = self.devices[device_key]
        
        # Memory threshold
        if stats['memory_available'] > 0:
            memory_usage_ratio = stats['memory_used'] / (stats['memory_used'] + stats['memory_available'])
            if memory_usage_ratio > self.config.memory_threshold:
                self.logger.warning(f"High memory usage on {device_key}: {memory_usage_ratio:.1%}")
        
        # Temperature threshold
        if stats.get('temperature') and stats['temperature'] > self.config.temperature_threshold:
            self.logger.warning(f"High temperature on {device_key}: {stats['temperature']:.1f}°C")
    
    def get_device(self, device_type: Optional[DeviceType] = None) -> Optional[str]:
        """Get the best available device."""
        if device_type:
            # Find specific device type
            for device_key, device_info in self.devices.items():
                if device_info.device_type == device_type and device_info.is_available:
                    return device_key
            return None
        
        # Return primary device
        return self.primary_device if self.devices[self.primary_device].is_available else None
    
    def get_torch_device(self, device_key: Optional[str] = None) -> torch.device:
        """Get PyTorch device object."""
        if device_key is None:
            device_key = self.primary_device
        
        if device_key is None:
            return torch.device('cpu')
        
        device_info = self.devices[device_key]
        
        if device_info.device_type == DeviceType.CUDA:
            return torch.device(f'cuda:{device_info.device_id}')
        elif device_info.device_type == DeviceType.MPS:
            return torch.device('mps')
        elif device_info.device_type == DeviceType.TPU:
            import torch_xla.core.xla_model as xm
            return xm.xla_device()
        else:
            return torch.device('cpu')
    
    def allocate_devices(self, num_devices: int) -> List[str]:
        """Allocate multiple devices for distributed computation."""
        available_devices = [key for key, info in self.devices.items() if info.is_available]
        
        # Sort by capability (prefer GPU over CPU)
        available_devices.sort(key=lambda x: (
            self.devices[x].device_type != DeviceType.CPU,
            self.devices[x].memory_total
        ), reverse=True)
        
        return available_devices[:num_devices]
    
    def get_device_stats(self, device_key: Optional[str] = None) -> Dict[str, Any]:
        """Get current device statistics."""
        if device_key is None:
            device_key = self.primary_device
        
        if device_key not in self.devices:
            return {}
        
        device_info = self.devices[device_key]
        recent_stats = self.device_stats_history.get(device_key, [{}])[-1] if self.device_stats_history.get(device_key) else {}
        
        return {
            'device_key': device_key,
            'device_type': device_info.device_type.value,
            'name': device_info.name,
            'memory_total': device_info.memory_total,
            'memory_available': device_info.memory_available,
            'utilization': device_info.utilization,
            'is_available': device_info.is_available,
            'tensor_cores': device_info.tensor_cores,
            'recent_stats': recent_stats
        }
    
    def get_all_device_stats(self) -> Dict[str, Any]:
        """Get statistics for all devices."""
        return {
            'primary_device': self.primary_device,
            'total_devices': len(self.devices),
            'devices': {key: self.get_device_stats(key) for key in self.devices.keys()},
            'optimization_settings': self.optimization_settings
        }
    
    def set_device_availability(self, device_key: str, available: bool):
        """Set device availability (for maintenance, etc.)."""
        if device_key in self.devices:
            self.devices[device_key].is_available = available
            self.logger.info(f"Set {device_key} availability to {available}")
    
    def stop_monitoring(self):
        """Stop device monitoring."""
        if self.monitoring_active:
            self.monitoring_active = False
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=5.0)
            self.logger.info("Stopped device monitoring")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.stop_monitoring()

