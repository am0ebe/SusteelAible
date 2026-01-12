"""
GPU Utilities Module
====================

Shared GPU management utilities for ML pipelines.

Usage:
    from gpu_utils import GPUManager, get_device, clear_gpu_memory
"""

from pathlib import Path
import gc
from typing import Optional, Dict

import torch


def get_device() -> torch.device:
    """Get the best available device (GPU or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def clear_gpu_memory():
    """Clear GPU memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_gpu_info() -> Optional[Dict]:
    """Get GPU information if available."""
    if not torch.cuda.is_available():
        return None
    return {
        "name": torch.cuda.get_device_name(0),
        "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        "device_count": torch.cuda.device_count(),
    }


class GPUManager:
    """
    Manages GPU resources for ML pipelines.

    Usage:
        gpu = GPUManager()
        print(gpu)  # Shows device info
        gpu.clear()  # Clears memory
        gpu.emergency_cleanup(models_dict)  # Full cleanup
    """

    def __init__(self):
        self.device = get_device()
        self.device_id = 0 if self.device.type == "cuda" else -1
        self.info = get_gpu_info()

    def __str__(self) -> str:
        if self.info:
            return f"GPU: {self.info['name']} ({self.info['total_memory_gb']:.1f}GB)"
        return f"Device: {self.device.type.upper()}"

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"

    def clear(self):
        """Clear GPU memory cache."""
        clear_gpu_memory()

    def emergency_cleanup(self, models: Optional[Dict] = None):
        """
        Aggressive cleanup for OOM situations.

        Args:
            models: Dictionary of loaded models to unload
        """
        if models:
            for name in list(models.keys()):
                try:
                    if hasattr(models[name], 'model'):
                        models[name].model.cpu()
                    del models[name]
                except Exception:
                    pass
            models.clear()

        gc.collect()

        # Move any remaining CUDA tensors to CPU
        if self.is_cuda:
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) and obj.is_cuda:
                        obj.data = obj.data.cpu()
                except Exception:
                    pass

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
