"""
Training utilities for early stopping, metrics tracking, and result saving
"""

import json
import os
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping mechanism to prevent overfitting"""

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.001,
        restore_best_weights: bool = True,
    ):
        """
        Initialize early stopping

        Args:
            patience: Number of epochs with no improvement to wait
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        self.best_loss = float("inf")
        self.counter = 0
        self.best_weights = None
        self.early_stop = False

    def __call__(self, val_loss: float, model: torch.nn.Module) -> bool:
        """
        Check if training should stop early

        Args:
            val_loss: Current validation loss
            model: Model to potentially save weights

        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            # Improvement found
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
            logger.info(f"Validation loss improved to {val_loss:.6f}")
        else:
            # No improvement
            self.counter += 1
            logger.info(f"No improvement for {self.counter}/{self.patience} epochs")

        if self.counter >= self.patience:
            self.early_stop = True
            logger.info(
                f"Early stopping triggered after {self.counter} epochs without improvement"
            )

            # Restore best weights if requested
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
                logger.info("Restored best model weights")

        return self.early_stop


class TrainingMetrics:
    """Track and save training metrics"""

    def __init__(self, save_dir: str):
        """Initialize metrics tracker"""
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = {
            "train_losses": [],
            "val_losses": [],
            "epochs": [],
            "learning_rates": [],
            "timestamps": [],
        }

        # Create training log file
        self.log_file = self.save_dir / "training_log.jsonl"

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        lr: float,
        additional_metrics: Optional[Dict] = None,
    ):
        """Log metrics for an epoch"""
        timestamp = datetime.now().isoformat()

        # Update metrics
        self.metrics["epochs"].append(epoch)
        self.metrics["train_losses"].append(train_loss)
        self.metrics["val_losses"].append(val_loss)
        self.metrics["learning_rates"].append(lr)
        self.metrics["timestamps"].append(timestamp)

        # Create log entry
        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": lr,
            "timestamp": timestamp,
        }

        if additional_metrics:
            log_entry.update(additional_metrics)

        # Append to log file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        logger.info(
            f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, lr={lr:.2e}"
        )

    def save_final_results(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        total_epochs: int,
        early_stopped: bool = False,
    ):
        """Save final training results and model"""

        # Save metrics summary
        summary = {
            "total_epochs": total_epochs,
            "best_val_loss": (
                min(self.metrics["val_losses"]) if self.metrics["val_losses"] else None
            ),
            "final_train_loss": (
                self.metrics["train_losses"][-1]
                if self.metrics["train_losses"]
                else None
            ),
            "early_stopped": early_stopped,
            "training_completed": datetime.now().isoformat(),
        }

        with open(self.save_dir / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Save complete metrics
        with open(self.save_dir / "metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2)

        # Save model and optimizer state
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": total_epochs,
                "metrics": self.metrics,
            },
            self.save_dir / "final_checkpoint.pt",
        )

        logger.info(f"Training results saved to {self.save_dir}")
        logger.info(f"Summary: {summary}")

        return summary


def create_train_val_split(
    dataset, val_ratio: float = 0.2, seed: int = 42
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Create train/validation split from dataset"""

    dataset_size = len(dataset)
    val_size = int(val_ratio * dataset_size)
    train_size = dataset_size - val_size

    # Create split with seed for reproducibility
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )

    logger.info(f"Dataset split: {train_size} train, {val_size} validation samples")
    return train_dataset, val_dataset


def evaluate_model(
    model: torch.nn.Module,
    val_dataset: torch.utils.data.Dataset,
    processor,
    device: str,
    task_type: str = "grounding",
) -> float:
    """Evaluate model on validation set"""

    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for i in range(len(val_dataset)):
            try:
                if task_type == "grounding":
                    inputs, target = val_dataset[i]
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    target = {k: v.to(device) for k, v in target.items()}

                    # Skip samples with invalid data
                    skip_sample = False
                    for k, v in inputs.items():
                        if torch.isnan(v).any() or torch.isinf(v).any():
                            skip_sample = True
                            break

                    # Validate target
                    for k, v in target.items():
                        if torch.isnan(v).any() or torch.isinf(v).any():
                            skip_sample = True
                            break

                    # Additional validation for GroundingDino
                    if 'boxes' in target and len(target['boxes']) == 0:
                        skip_sample = True

                    if 'class_labels' in target:
                        labels = target['class_labels']
                        if not torch.all((labels >= 0) & (labels <= 1)):
                            skip_sample = True

                    if skip_sample:
                        continue

                    outputs = model(**inputs, labels=[target])

                elif task_type == "vqa":
                    img, quest, ans = val_dataset[i]
                    inputs = processor(images=img, text=quest, return_tensors="pt").to(
                        device
                    )
                    labels = processor(text=ans, return_tensors="pt").input_ids.to(
                        device
                    )
                    outputs = model(**inputs, labels=labels)

                total_loss += outputs.loss.item()
                num_batches += 1

            except Exception as e:
                logger.warning(f"Error evaluating sample {i}: {e}")
                continue

    avg_loss = total_loss / max(num_batches, 1)
    model.train()  # Set back to training mode

    return avg_loss


def setup_training_directory(model_name: str) -> str:
    """Setup training directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_dir = f"./training_results/{model_name}_{timestamp}"

    os.makedirs(training_dir, exist_ok=True)
    logger.info(f"Training directory created: {training_dir}")

    return training_dir


def setup_pretrained_directory(model_name: str) -> str:
    """Setup pretrained model directory"""
    pretrained_dir = f"./pretrained/{model_name}"

    os.makedirs(pretrained_dir, exist_ok=True)
    logger.info(f"Pretrained directory created: {pretrained_dir}")

    return pretrained_dir


def save_pretrained_model(model, processor, model_name: str, training_summary: dict) -> str:
    """Save model to pretrained directory with metadata"""
    pretrained_path = setup_pretrained_directory(model_name)

    # Save model and processor
    model.save_pretrained(pretrained_path)
    processor.save_pretrained(pretrained_path)

    # Save training metadata
    metadata = {
        'model_name': model_name,
        'saved_at': datetime.now().isoformat(),
        'training_summary': training_summary
    }

    metadata_path = os.path.join(pretrained_path, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Pretrained model saved to: {pretrained_path}")
    return pretrained_path
