"""
Device detection and hardware acceleration utilities.
"""
import torch
import logging

logger = logging.getLogger(__name__)


def get_device():
    """
    Automatically detect and return the best available device.

    Priority order:
    1. CUDA (if available)
    2. MPS (Apple Silicon - if available)
    3. CPU (fallback)

    Returns:
        torch.device: The best available device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        return device

    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using MPS (Apple Silicon) acceleration")
        return device

    else:
        device = torch.device('cpu')
        logger.info("Using CPU (no hardware acceleration available)")
        return device


def get_device_info():
    """
    Get detailed information about the current device configuration.

    Returns:
        dict: Device information including type, name, memory, etc.
    """
    device = get_device()
    info = {
        'device': str(device),
        'type': device.type,
        'available_devices': []
    }

    # CUDA information
    if torch.cuda.is_available():
        info['cuda_available'] = True
        info['cuda_device_count'] = torch.cuda.device_count()
        info['cuda_device_name'] = torch.cuda.get_device_name()
        if torch.cuda.is_available():
            # Get memory info for current device
            try:
                memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                info['cuda_memory'] = {
                    'allocated_gb': round(memory_allocated, 2),
                    'reserved_gb': round(memory_reserved, 2),
                    'total_gb': round(memory_total, 2)
                }
            except Exception as e:
                logger.warning(f"Could not get CUDA memory info: {e}")
        info['available_devices'].append('cuda')
    else:
        info['cuda_available'] = False

    # MPS information
    if torch.backends.mps.is_available():
        info['mps_available'] = True
        info['available_devices'].append('mps')
    else:
        info['mps_available'] = False

    # CPU is always available
    info['available_devices'].append('cpu')

    return info


def setup_device_optimization(device):
    """
    Setup device-specific optimizations.

    Args:
        device (torch.device): The device to optimize for
    """
    if device.type == 'cuda':
        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        logger.info("Enabled CUDA optimizations (cuDNN benchmark)")

    elif device.type == 'mps':
        # MPS-specific optimizations
        logger.info("Using MPS optimizations")

    else:
        logger.info("Using CPU optimizations")


def check_device_compatibility(model, device):
    """
    Check if the model is compatible with the specified device.

    Args:
        model: PyTorch model
        device (torch.device): Target device

    Returns:
        bool: True if compatible, False otherwise
    """
    try:
        # Try to move model to device
        model.to(device)

        # Create a dummy input tensor and test forward pass
        dummy_input = torch.randn(1, 10).to(device)  # Adjust size as needed
        with torch.no_grad():
            _ = model(dummy_input)

        logger.info(f"Model is compatible with {device}")
        return True

    except Exception as e:
        logger.error(f"Model compatibility check failed for {device}: {e}")
        return False


def get_optimal_batch_size(device, model_size_mb=100):
    """
    Estimate optimal batch size based on available memory and model size.

    Args:
        device (torch.device): Target device
        model_size_mb (int): Estimated model size in MB

    Returns:
        int: Suggested batch size
    """
    if device.type == 'cuda' and torch.cuda.is_available():
        try:
            # Get available GPU memory
            total_memory = torch.cuda.get_device_properties(0).total_memory
            available_memory = total_memory - torch.cuda.memory_reserved()
            available_gb = available_memory / 1024**3

            # Estimate batch size (conservative approach)
            # Reserve memory for model, gradients, and activations
            safety_factor = 0.7  # Use 70% of available memory
            memory_per_sample_mb = model_size_mb * 4  # Model + gradients + activations

            estimated_batch_size = int((available_gb * 1024 * safety_factor) / memory_per_sample_mb)
            batch_size = max(1, min(estimated_batch_size, 128))  # Cap at 128

            logger.info(f"Estimated optimal batch size for CUDA: {batch_size}")
            return batch_size

        except Exception as e:
            logger.warning(f"Could not estimate CUDA batch size: {e}")
            return 32

    elif device.type == 'mps':
        # MPS typically has good memory management, but be conservative
        logger.info("Estimated optimal batch size for MPS: 64")
        return 64

    else:
        # CPU - smaller batches are usually better
        logger.info("Estimated optimal batch size for CPU: 16")
        return 16


if __name__ == "__main__":
    # Test device detection
    import json

    logging.basicConfig(level=logging.INFO)

    print("Device Detection Test")
    print("=" * 50)

    device = get_device()
    print(f"Selected device: {device}")

    info = get_device_info()
    print(f"\nDevice Information:")
    print(json.dumps(info, indent=2))

    setup_device_optimization(device)

    batch_size = get_optimal_batch_size(device)
    print(f"\nOptimal batch size: {batch_size}")