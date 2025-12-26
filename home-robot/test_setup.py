"""Test script to verify home robot setup"""
import torch
from device_utils import detect_optimal_device, get_device_info
import logging

logging.basicConfig(level=logging.INFO)

def test_device_detection():
    """Test device detection functionality"""
    print("Testing device detection...")
    device = detect_optimal_device()
    print(f"Detected device: {device}")

    device_info = get_device_info()
    print("Device information:")
    for key, value in device_info.items():
        print(f"  {key}: {value}")

def test_imports():
    """Test critical imports"""
    print("\nTesting imports...")
    try:
        from transformers import GroundingDinoProcessor, Blip2Processor
        print("✓ Transformers imports successful")

        # Test processor loading (doesn't require local model files)
        print("✓ Attempting to load processors...")
        dino_proc = GroundingDinoProcessor.from_pretrained(
            "IDEA-Research/grounding-dino-tiny"
        )
        print("✓ GroundingDino processor loaded")

        blip_proc = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        print("✓ BLIP2 processor loaded")

        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Model loading error: {e}")
        print("Note: This may be expected if models aren't downloaded yet")
        return False

def test_basic_functionality():
    """Test basic torch functionality"""
    print("\nTesting basic PyTorch functionality...")
    try:
        # Test tensor creation
        device = detect_optimal_device()
        tensor = torch.randn(2, 3).to(device)
        print(f"✓ Created tensor on {device}: shape {tensor.shape}")

        # Test basic operations
        result = tensor * 2
        print(f"✓ Basic tensor operations working")

        return True
    except Exception as e:
        print(f"✗ PyTorch functionality error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Home Robot System Setup Verification")
    print("=" * 50)

    # Run all tests
    device_ok = test_device_detection()
    imports_ok = test_imports()
    torch_ok = test_basic_functionality()

    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Device Detection: {'✓ PASS' if device_ok else '✗ FAIL'}")
    print(f"Import Tests: {'✓ PASS' if imports_ok else '✗ FAIL'}")
    print(f"PyTorch Tests: {'✓ PASS' if torch_ok else '✗ FAIL'}")

    if all([device_ok, torch_ok]):
        print("\n🎉 Setup verification completed successfully!")
        print("You can now use the HomeRobotSystem.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        print("You may need to install dependencies or check your environment.")