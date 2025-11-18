"""
Example usage of the CNN image classification project
This script demonstrates how to use the main components
"""

import torch
from src.utils.device import get_device, get_device_info
from src.models.resnet import create_model, count_parameters


def main():
    """Demonstrate project functionality"""
    print("🚀 CNN Image Classification Project Demo")
    print("=" * 50)

    # Test device detection
    print("\n📱 Device Detection:")
    device = get_device()
    device_info = get_device_info(device)

    for key, value in device_info.items():
        print(f"  {key}: {value}")

    # Test model creation
    print("\n🧠 Model Creation:")
    model = create_model(
        num_classes=10,
        pretrained=True,
        dropout_rate=0.5,
        device=device
    )

    total_params, trainable_params = count_parameters(model)
    print(f"  Model: ResNet34-based classifier")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Device: {device}")

    # Test forward pass with dummy data
    print("\n🔮 Forward Pass Test:")
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Number of classes: {output.shape[1]}")

    # Check predictions
    probabilities = torch.softmax(output, dim=1)
    predictions = torch.argmax(probabilities, dim=1)

    print(f"  Predictions: {predictions.cpu().numpy()}")
    print(f"  Max probabilities: {torch.max(probabilities, dim=1)[0].cpu().numpy()}")

    print("\n✅ All components working correctly!")
    print("\n📚 Next steps:")
    print("  1. Run 'uv run train' to start training")
    print("  2. Run 'uv run evaluate --checkpoint models/best.pth' after training")
    print("  3. Monitor training with 'tensorboard --logdir logs/tensorboard'")


if __name__ == '__main__':
    main()