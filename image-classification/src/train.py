"""
Training script for CNN image classification
Supports multi-device training with auto-detection (CUDA, MPS, CPU)
"""

import os
import sys
import time
import yaml
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add src directory to path for direct imports
src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, src_dir)

from data.dataset import create_data_loaders
from models.resnet import create_model, count_parameters
from utils.device import get_device, get_device_info, clear_memory, set_device_settings
from utils.metrics import AverageMeter, MetricsCalculator


class Trainer:
    """Main trainer class for CNN image classification"""

    def __init__(self, config_path: str):
        """
        Initialize trainer with configuration

        Args:
            config_path: Path to configuration YAML file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Set random seeds for reproducibility
        self._set_seeds()

        # Setup device
        self.device = self._setup_device()

        # Create directories
        self._create_directories()

        # Setup logging
        self.writer = None
        if self.config['logging']['tensorboard']:
            self.writer = SummaryWriter(self.config['logging']['tensorboard_log_dir'])

        # Initialize metrics
        self.best_accuracy = 0.0
        self.global_step = 0

    def _set_seeds(self):
        """Set random seeds for reproducibility"""
        seed = self.config.get('seed', 42)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if self.config.get('deterministic', False):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _setup_device(self) -> torch.device:
        """Setup and configure device"""
        device_config = self.config.get('device', 'auto')

        if device_config == 'auto':
            device = get_device()
        else:
            device = torch.device(device_config)

        # Print device information
        device_info = get_device_info(device)
        print(f"\nDevice Information:")
        for key, value in device_info.items():
            print(f"  {key}: {value}")

        # Configure device settings
        set_device_settings(device)

        return device

    def _create_directories(self):
        """Create necessary directories"""
        dirs = [
            self.config['logging']['checkpoint_dir'],
            self.config['logging']['log_dir'],
            self.config['logging']['results_dir']
        ]

        if self.config['logging']['tensorboard']:
            dirs.append(self.config['logging']['tensorboard_log_dir'])

        for directory in dirs:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def _create_data_loaders(self):
        """Create train and test data loaders"""
        print("\nCreating data loaders...")
        data_config = self.config['data']

        train_loader, test_loader = create_data_loaders(
            train_file=data_config['train_file'],
            test_file=data_config['test_file'],
            data_root=data_config['data_root'],
            batch_size=data_config['batch_size'],
            num_workers=data_config['num_workers'],
            image_size=data_config['image_size']
        )

        # Print dataset information
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")
        print(f"Batch size: {data_config['batch_size']}")
        print(f"Training batches: {len(train_loader)}")
        print(f"Test batches: {len(test_loader)}")

        # Print class distribution
        train_class_counts = train_loader.dataset.get_class_counts()
        print(f"\nClass distribution in training set:")
        for class_id, count in sorted(train_class_counts.items()):
            print(f"  Class {class_id}: {count} samples")

        return train_loader, test_loader

    def _create_model(self):
        """Create and initialize model"""
        print("\nCreating model...")
        model_config = self.config['model']

        model = create_model(
            num_classes=self.config['data']['num_classes'],
            pretrained=model_config['pretrained'],
            dropout_rate=model_config['dropout_rate'],
            freeze_backbone=model_config['freeze_backbone'],
            device=self.device
        )

        # Print model information
        total_params, trainable_params = count_parameters(model)
        print(f"Model: ResNet34-based classifier")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")

        return model

    def _create_optimizer(self, model):
        """Create optimizer"""
        optimizer_config = self.config['optimizer']
        training_config = self.config['training']

        if optimizer_config['type'].lower() == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=training_config['learning_rate'],
                weight_decay=training_config['weight_decay'],
                betas=(optimizer_config['beta1'], optimizer_config['beta2']),
                eps=optimizer_config['eps']
            )
        elif optimizer_config['type'].lower() == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=training_config['learning_rate'],
                weight_decay=training_config['weight_decay'],
                betas=(optimizer_config['beta1'], optimizer_config['beta2']),
                eps=optimizer_config['eps']
            )
        elif optimizer_config['type'].lower() == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=training_config['learning_rate'],
                weight_decay=training_config['weight_decay'],
                momentum=optimizer_config['momentum']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_config['type']}")

        print(f"Optimizer: {optimizer_config['type'].upper()}")
        print(f"Learning rate: {training_config['learning_rate']}")
        print(f"Weight decay: {training_config['weight_decay']}")

        return optimizer

    def _create_scheduler(self, optimizer):
        """Create learning rate scheduler"""
        training_config = self.config['training']
        scheduler_type = training_config['scheduler']

        if scheduler_type == 'step':
            scheduler = StepLR(
                optimizer,
                step_size=training_config['step_size'],
                gamma=training_config['gamma']
            )
        elif scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=training_config['epochs']
            )
        elif scheduler_type == 'plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=training_config['factor'],
                patience=training_config['patience'],
                verbose=True
            )
        else:
            scheduler = None

        if scheduler:
            print(f"Scheduler: {scheduler_type}")

        return scheduler

    def _create_criterion(self):
        """Create loss criterion"""
        loss_config = self.config['loss']

        if loss_config['type'] == 'cross_entropy':
            label_smoothing = loss_config.get('label_smoothing', 0.0)
            criterion = nn.CrossEntropyLoss(
                weight=loss_config.get('class_weights'),
                label_smoothing=label_smoothing
            )
        else:
            raise ValueError(f"Unsupported loss type: {loss_config['type']}")

        criterion = criterion.to(self.device)
        print(f"Loss function: {loss_config['type']}")

        return criterion

    def train_epoch(self, model, train_loader, criterion, optimizer, epoch):
        """Train for one epoch"""
        model.train()

        # Metrics for tracking
        losses = AverageMeter('Loss', ':.6f')
        accuracies = AverageMeter('Acc@1', ':6.2f')

        # Progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config["training"]["epochs"]}')

        # Mixed precision training
        scaler = torch.cuda.amp.GradScaler() if self.config['training'].get('mixed_precision', False) else None

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass with mixed precision
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                # Backward pass
                scaler.scale(loss).backward()

                # Gradient clipping
                if self.config['training'].get('gradient_clipping', False):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        self.config['training']['max_grad_norm']
                    )

                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Backward pass
                loss.backward()

                # Gradient clipping
                if self.config['training'].get('gradient_clipping', False):
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        self.config['training']['max_grad_norm']
                    )

                optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == targets).float().mean().item() * 100

            # Update metrics
            losses.update(loss.item(), inputs.size(0))
            accuracies.update(accuracy, inputs.size(0))

            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Acc': f'{accuracies.avg:.2f}%'
            })

            # Log to tensorboard
            if self.writer and batch_idx % self.config['logging']['log_interval'] == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), self.global_step)
                self.writer.add_scalar('Train/Accuracy', accuracy, self.global_step)
                self.writer.add_scalar('Train/Learning_Rate', optimizer.param_groups[0]['lr'], self.global_step)

            self.global_step += 1

        return losses.avg, accuracies.avg

    def validate(self, model, test_loader, criterion):
        """Validate the model"""
        model.eval()

        losses = AverageMeter('Loss', ':.6f')
        metrics_calc = MetricsCalculator(
            self.config['data']['num_classes'],
            [f"Class {i+1}" for i in range(self.config['data']['num_classes'])]
        )

        with torch.no_grad():
            pbar = tqdm(test_loader, desc='Validation')

            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Update metrics
                losses.update(loss.item(), inputs.size(0))
                metrics_calc.update(outputs, targets)

                # Update progress bar
                pbar.set_postfix({'Loss': f'{losses.avg:.4f}'})

        # Calculate final metrics
        metrics = metrics_calc.compute_metrics()
        accuracy = metrics.get('accuracy', 0.0) * 100

        return losses.avg, accuracy, metrics

    def save_checkpoint(self, model, optimizer, scheduler, epoch, accuracy, is_best=False):
        """Save model checkpoint"""
        if not self.config['logging']['save_model']:
            return

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'accuracy': accuracy,
            'config': self.config
        }

        checkpoint_dir = Path(self.config['logging']['checkpoint_dir'])

        # Save latest checkpoint
        if self.config['logging']['save_last']:
            torch.save(checkpoint, checkpoint_dir / 'latest.pth')

        # Save best checkpoint
        if is_best and self.config['logging']['save_best_only']:
            torch.save(checkpoint, checkpoint_dir / 'best.pth')
            print(f"New best model saved with accuracy: {accuracy:.2f}%")

    def train(self):
        """Main training loop"""
        print("Starting training...")
        print(f"Configuration: {self.config}")

        # Create data loaders
        train_loader, test_loader = self._create_data_loaders()

        # Create model
        model = self._create_model()

        # Create optimizer, scheduler, and criterion
        optimizer = self._create_optimizer(model)
        scheduler = self._create_scheduler(optimizer)
        criterion = self._create_criterion()

        # Training loop
        early_stopping_counter = 0
        start_time = time.time()

        for epoch in range(self.config['training']['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['training']['epochs']}")

            # Train for one epoch
            train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer, epoch)

            # Validate
            val_loss, val_acc, val_metrics = self.validate(model, test_loader, criterion)

            # Update learning rate scheduler
            if scheduler:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_acc)
                else:
                    scheduler.step()

            # Log epoch results
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Log to tensorboard
            if self.writer:
                self.writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
                self.writer.add_scalar('Epoch/Train_Accuracy', train_acc, epoch)
                self.writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
                self.writer.add_scalar('Epoch/Val_Accuracy', val_acc, epoch)

            # Check for best model
            is_best = val_acc > self.best_accuracy
            if is_best:
                self.best_accuracy = val_acc
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            # Save checkpoint
            self.save_checkpoint(model, optimizer, scheduler, epoch, val_acc, is_best)

            # Early stopping
            if (self.config['training'].get('early_stopping', False) and
                early_stopping_counter >= self.config['training']['early_stopping_patience']):
                print(f"\nEarly stopping triggered after {early_stopping_counter} epochs without improvement")
                break

            # Clear memory
            clear_memory(self.device)

        # Training completed
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f} seconds")
        print(f"Best validation accuracy: {self.best_accuracy:.2f}%")

        # Close tensorboard writer
        if self.writer:
            self.writer.close()

        return model


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train CNN for image classification')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    args = parser.parse_args()

    # Create trainer and start training
    trainer = Trainer(args.config)
    model = trainer.train()

    print(f"\nTraining completed successfully!")
    print(f"Best accuracy achieved: {trainer.best_accuracy:.2f}%")


if __name__ == '__main__':
    main()