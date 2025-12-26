import torch
import platform
import logging

logger = logging.getLogger(__name__)

def detect_optimal_device():
    """
    Auto-detect the best available device for PyTorch operations.

    Priority order:
    1. CUDA (NVIDIA GPU) - if available
    2. MPS (Apple Silicon GPU) - if available on macOS
    3. CPU - fallback option

    Returns:
        str: Device string ("cuda", "mps", or "cpu")
    """
    # Check for CUDA availability
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Using CUDA device: {gpu_name}")
        return device

    # Check for MPS (Apple Silicon) availability
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        if platform.system() == "Darwin":  # macOS
            device = "mps"
            logger.info("Using MPS (Apple Silicon GPU) device")
            return device

    # Fallback to CPU
    device = "cpu"
    logger.info("Using CPU device")
    return device

def get_device_info():
    """
    Get detailed information about the current device setup.

    Returns:
        dict: Device information including type, memory, etc.
    """
    device = detect_optimal_device()
    info = {
        "device": device,
        "torch_version": torch.__version__,
        "platform": platform.system(),
    }

    if device == "cuda":
        info.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB",
            "cuda_version": torch.version.cuda,
        })
    elif device == "mps":
        info.update({
            "mps_available": torch.backends.mps.is_available(),
            "mps_built": torch.backends.mps.is_built(),
        })

    return info