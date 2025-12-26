import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from torch import optim
from peft import LoraConfig, get_peft_model
import logging
from tqdm import tqdm

# Import from parent directory
from data.nyu_depth_v2 import NYURobotDataset
from device_utils import detect_optimal_device
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


def train_blip2_vqa(
    max_epochs=10,
    patience=5,
    val_ratio=0.2,
    save_model_name="robot_blip_lora",
):
    """
    Train BLIP2 model with LoRA for VQA task with early stopping

    Args:
        mat_path: Path to the dataset
        max_epochs: Maximum number of training epochs
        patience: Early stopping patience
        val_ratio: Validation split ratio
        save_model_name: Name for saved model
    """

    # Setup device and training directory
    device = detect_optimal_device()
    training_dir = setup_training_directory(save_model_name)

    logger.info("=" * 60)
    logger.info("BLIP2 VQA Training with LoRA")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Training directory: {training_dir}")

    # Load processor and model
    logger.info("Loading BLIP2 processor and model...")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        device_map="auto" if device == "cuda" else None,
        load_in_8bit=True if device == "cuda" else False,
    )

    # Move to device if not using device_map
    if device != "cuda":
        model = model.to(device)

    # Configure LoRA
    logger.info("Setting up LoRA configuration...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    # Load dataset and create train/val split
    logger.info("Loading dataset and creating train/val split...")
    full_dataset = NYURobotDataset(processor, task="vqa")
    train_dataset, val_dataset = create_train_val_split(
        full_dataset, val_ratio=val_ratio
    )

    # Setup optimizer and training components
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    early_stopping = EarlyStopping(patience=patience, min_delta=0.001)
    metrics = TrainingMetrics(training_dir)

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
                img, quest, ans = train_dataset[i]

                # Prepare inputs and move to device
                inputs = processor(images=img, text=quest, return_tensors="pt").to(
                    device
                )
                labels = processor(text=ans, return_tensors="pt").input_ids.to(device)

                # Forward pass
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss

                # Backward pass
                loss.backward()
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
            model, val_dataset, processor, device, task_type="vqa"
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

    # Save final LoRA model to pretrained directory (for inference)
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

    parser = argparse.ArgumentParser(description="Train BLIP2 VQA model")
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum epochs")
    parser.add_argument(
        "--patience", type=int, default=5, help="Early stopping patience"
    )
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation ratio")

    args = parser.parse_args()

    result = train_blip2_vqa(
        max_epochs=args.max_epochs,
        patience=args.patience,
        val_ratio=args.val_ratio,
    )

    print(f"\nTraining completed!")
    print(f"Training model saved to: {result['training_save_path']}")
    print(f"Pretrained model saved to: {result['pretrained_path']}")
