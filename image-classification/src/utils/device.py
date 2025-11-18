"""
Device detection and management utilities
Automatically detects and configures the best available device (CUDA, MPS, CPU)
"""

import torch
import platform


def get_device() -> torch.device:
    """
    Automatically detect and return the best available device

    Returns:
        torch.device: The best available device (cuda, mps, or cpu)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    elif torch.backends.mps.is_available() and platform.system() == "Darwin":
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")

    return device


def get_device_info(device: torch.device) -> dict:
    """
    Get detailed information about the device

    Args:
        device: PyTorch device

    Returns:
        Dictionary containing device information
    """
    info = {
        "device_type": device.type,
        "device_name": str(device)
    }

    if device.type == "cuda":
        info.update({
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device),
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "memory_allocated": torch.cuda.memory_allocated(device),
            "memory_reserved": torch.cuda.memory_reserved(device),
            "max_memory_allocated": torch.cuda.max_memory_allocated(device),
            "total_memory": torch.cuda.get_device_properties(device).total_memory
        })
    elif device.type == "mps":
        info.update({
            "mps_available": torch.backends.mps.is_available(),
            "mps_built": torch.backends.mps.is_built()
        })

    return info


def clear_memory(device: torch.device):
    """
    Clear GPU memory cache if using CUDA

    Args:
        device: PyTorch device
    """
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print("Cleared CUDA memory cache")


def set_device_settings(device: torch.device):
    """
    Configure device-specific settings for optimal performance

    Args:
        device: PyTorch device
    """
    if device.type == "cuda":
        # Enable cuDNN benchmark for faster training with fixed input sizes
        torch.backends.cudnn.benchmark = True
        # Enable cuDNN deterministic mode for reproducible results (if needed)
        # torch.backends.cudnn.deterministic = True
        print("Enabled cuDNN benchmark mode")

    # Set number of threads for CPU operations
    if device.type == "cpu":
        # Use all available CPU cores
        torch.set_num_threads(torch.get_num_threads())
        print(f"Using {torch.get_num_threads()} CPU threads")