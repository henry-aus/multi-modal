import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import GroundingDinoProcessor, GroundingDinoForObjectDetection
import torch.optim as optim
import logging
from tqdm import tqdm

# Import from parent directory
from device_utils import detect_optimal_device
from data.nyu_depth_v2 import NYURobotDataset
from train.training_utils import (
    EarlyStopping,
    TrainingMetrics,
    create_train_val_split,
    evaluate_model,
    setup_training_directory,
    save_pretrained_model,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_grounding_dino(
    max_epochs=15, patience=7, val_ratio=0.2, save_model_name="robot_dino_final"
):
    """
    Train GroundingDino model with early stopping

    Args:
        mat_path: Path to the dataset
        max_epochs: Maximum number of training epochs
        patience: Early stopping patience
        val_ratio: Validation split ratio
        save_model_name: Name for saved model
    """

    # Setup device and training directory
    # Note: GroundingDino has issues with MPS due to unimplemented grid_sampler_2d_backward
    # Force CPU for better compatibility
    base_device = detect_optimal_device()
    if base_device.startswith('mps'):
        device = 'cpu'
        logger.warning("Detected MPS device, but GroundingDino has compatibility issues with MPS.")
        logger.warning("Falling back to CPU for stable training.")
    else:
        device = base_device

    training_dir = setup_training_directory(save_model_name)

    logger.info("=" * 60)
    logger.info("GroundingDino Object Detection Training")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Training directory: {training_dir}")

    # Load processor and model
    logger.info("Loading GroundingDino processor and model...")
    processor = GroundingDinoProcessor.from_pretrained(
        "IDEA-Research/grounding-dino-tiny"
    )
    model = GroundingDinoForObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-tiny"
    ).to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Load dataset and create train/val split
    logger.info("Loading dataset and creating train/val split...")
    full_dataset = NYURobotDataset(processor, task="grounding")
    train_dataset, val_dataset = create_train_val_split(
        full_dataset, val_ratio=val_ratio
    )

    # Setup optimizer and training components with more conservative learning rate
    optimizer = optim.AdamW(model.parameters(), lr=1e-6, weight_decay=1e-4)
    early_stopping = EarlyStopping(patience=patience, min_delta=0.001)
    metrics = TrainingMetrics(training_dir)

    # Check model parameters for any initial NaN values
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            logger.error(f"NaN detected in model parameter '{name}' during initialization!")
        if torch.isinf(param).any():
            logger.error(f"Inf detected in model parameter '{name}' during initialization!")

    logger.info(f"Model initialized successfully. Learning rate: {optimizer.param_groups[0]['lr']}")

    # Training loop
    logger.info("Starting training...")
    model.train()

    for epoch in range(max_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{max_epochs}")
        logger.info("-" * 40)

        # Training phase
        train_loss = 0.0
        train_steps = 0

        # Use tqdm for progress bar
        pbar = tqdm(range(len(train_dataset)), desc=f"Training Epoch {epoch + 1}")

        for i in pbar:
            try:
                inputs, target = train_dataset[i]

                # Move inputs to device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Move target tensors to device
                target = {k: v.to(device) for k, v in target.items()}

                # Basic tensor validation
                skip_sample = False
                for k, v in inputs.items():
                    if torch.isnan(v).any() or torch.isinf(v).any():
                        logger.warning(f"Invalid values in input '{k}' for sample {i}")
                        skip_sample = True
                        break

                # Validate target (should contain 'boxes' and 'class_labels' for GroundingDino)
                for k, v in target.items():
                    if torch.isnan(v).any() or torch.isinf(v).any():
                        logger.warning(f"Invalid values in target '{k}' for sample {i}")
                        skip_sample = True
                        break

                # Additional validation for GroundingDino format
                if 'boxes' in target and len(target['boxes']) == 0:
                    logger.warning(f"Empty boxes in target for sample {i}")
                    skip_sample = True

                if 'class_labels' in target:
                    labels = target['class_labels']
                    # Check that all labels are 0 or 1 (GroundingDino binary format)
                    if not torch.all((labels >= 0) & (labels <= 1)):
                        logger.warning(f"Invalid class labels for sample {i}: {labels.tolist()}")
                        skip_sample = True

                if skip_sample:
                    continue

                # Forward pass - GroundingDino expects labels as a list
                outputs = model(**inputs, labels=[target])
                loss = outputs.loss

                # Check for invalid loss
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"Invalid loss detected for sample {i}: {loss.item()}")
                    continue

                # Backward pass
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                optimizer.zero_grad()

                # Update metrics
                train_loss += loss.item()
                train_steps += 1

                # Update progress bar
                pbar.set_postfix({"loss": f"{loss.item():.6f}"})

            except Exception as e:
                logger.warning(f"Error processing sample {i}: {e}")
                continue

        # Calculate average training loss
        avg_train_loss = train_loss / max(train_steps, 1)

        # Validation phase
        logger.info("Running validation...")
        avg_val_loss = evaluate_model(
            model, val_dataset, processor, device, task_type="grounding"
        )

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        # Log epoch metrics
        metrics.log_epoch(
            epoch + 1,
            avg_train_loss,
            avg_val_loss,
            current_lr,
            additional_metrics={"train_steps": train_steps},
        )

        # Check early stopping
        if early_stopping(avg_val_loss, model):
            logger.info("Early stopping triggered!")
            break

    # Training completed
    final_epoch = epoch + 1
    early_stopped = early_stopping.early_stop

    logger.info("=" * 60)
    logger.info("Training completed!")
    logger.info(f"Total epochs: {final_epoch}")
    logger.info(f"Early stopped: {early_stopped}")
    logger.info("=" * 60)

    # Save training results
    summary = metrics.save_final_results(model, optimizer, final_epoch, early_stopped)

    # Save model to training results (for debugging/analysis)
    training_model_path = os.path.join(training_dir, save_model_name)
    model.save_pretrained(training_model_path)
    processor.save_pretrained(training_model_path)

    # Save final model to pretrained directory (for inference)
    pretrained_path = save_pretrained_model(model, processor, save_model_name, summary)

    logger.info(f"Training model saved to: {training_model_path}")
    logger.info(f"Pretrained model saved to: {pretrained_path}")

    return {
        "model": model,
        "processor": processor,
        "training_summary": summary,
        "training_save_path": training_model_path,
        "pretrained_path": pretrained_path,
    }


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Train GroundingDino model")
    parser.add_argument("--max_epochs", type=int, default=15, help="Maximum epochs")
    parser.add_argument(
        "--patience", type=int, default=7, help="Early stopping patience"
    )
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation ratio")

    args = parser.parse_args()

    result = train_grounding_dino(
        max_epochs=args.max_epochs, patience=args.patience, val_ratio=args.val_ratio
    )

    print(f"\nTraining completed!")
    print(f"Training model saved to: {result['training_save_path']}")
    print(f"Pretrained model saved to: {result['pretrained_path']}")
