"""
Training pipeline for neural network models.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Tuple, Optional, Callable
import numpy as np
from tqdm import tqdm
import logging
import os
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

logger = logging.getLogger(__name__)


class Trainer:
    """
    Training pipeline for PyTorch neural network models.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        optimizer_name: str = "adam",
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        scheduler_name: Optional[str] = "step",
        scheduler_params: Optional[Dict[str, Any]] = None,
        gradient_clip_norm: float = 1.0,
        early_stopping_patience: int = 10,
        early_stopping_metric: str = "val_loss",
        min_delta: float = 0.001,
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        Initialize Trainer.

        Args:
            model (nn.Module): PyTorch model to train
            device (torch.device): Device to train on
            optimizer_name (str): Optimizer name ('adam', 'sgd', 'rmsprop')
            learning_rate (float): Learning rate
            weight_decay (float): Weight decay for regularization
            scheduler_name (str, optional): Scheduler name ('step', 'cosine', 'plateau')
            scheduler_params (Dict[str, Any], optional): Scheduler parameters
            gradient_clip_norm (float): Gradient clipping norm
            early_stopping_patience (int): Early stopping patience
            early_stopping_metric (str): Metric to monitor for early stopping
            min_delta (float): Minimum change for early stopping
            class_weights (torch.Tensor, optional): Class weights for imbalanced data
        """
        self.model = model.to(device)
        self.device = device
        self.gradient_clip_norm = gradient_clip_norm
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_metric = early_stopping_metric
        self.min_delta = min_delta

        # Initialize optimizer
        self.optimizer = self._create_optimizer(optimizer_name, learning_rate, weight_decay)

        # Initialize scheduler
        if scheduler_name:
            self.scheduler = self._create_scheduler(scheduler_name, scheduler_params or {})
        else:
            self.scheduler = None

        # Loss function
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Training history
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }

        # Early stopping state
        self.best_metric = float('inf') if 'loss' in early_stopping_metric else 0.0
        self.patience_counter = 0
        self.best_model_state = None

        logger.info(f"Trainer initialized:")
        logger.info(f"  Model: {model.__class__.__name__}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Optimizer: {optimizer_name}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Early stopping: {early_stopping_metric} (patience={early_stopping_patience})")

    def _create_optimizer(self, name: str, lr: float, weight_decay: float) -> optim.Optimizer:
        """Create optimizer."""
        if name.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif name.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        elif name.lower() == 'rmsprop':
            return optim.RMSprop(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {name}")

    def _create_scheduler(self, name: str, params: Dict[str, Any]) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        if name.lower() == 'step':
            step_size = params.get('step_size', 30)
            gamma = params.get('gamma', 0.1)
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif name.lower() == 'cosine':
            T_max = params.get('T_max', 100)
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max)
        elif name.lower() == 'plateau':
            patience = params.get('patience', 5)
            factor = params.get('factor', 0.5)
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=patience, factor=factor
            )
        else:
            raise ValueError(f"Unsupported scheduler: {name}")

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader (DataLoader): Training data loader

        Returns:
            Dict[str, float]: Training metrics
        """
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_labels = []

        for batch in tqdm(train_loader, desc="Training", leave=False):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)

            self.optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_predictions)

        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }

    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate for one epoch.

        Args:
            val_loader (DataLoader): Validation data loader

        Returns:
            Dict[str, float]: Validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)

                # Accumulate metrics
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'predictions': all_predictions,
            'true_labels': all_labels
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100,
        verbose: bool = True,
        save_best_model: bool = True,
        model_save_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            num_epochs (int): Number of epochs to train
            verbose (bool): Whether to print progress
            save_best_model (bool): Whether to save the best model
            model_save_path (str, optional): Path to save the best model

        Returns:
            Dict[str, List[float]]: Training history
        """
        logger.info(f"Starting training for {num_epochs} epochs...")

        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start_time = time.time()

            # Train epoch
            train_metrics = self.train_epoch(train_loader)

            # Validate epoch
            val_metrics = self.validate_epoch(val_loader)

            # Update learning rate scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()

            # Record metrics
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_accuracy'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])

            # Check early stopping
            current_metric = val_metrics[self.early_stopping_metric.replace('val_', '')]
            improved = self._check_improvement(current_metric)

            if improved:
                self.best_metric = current_metric
                self.patience_counter = 0
                if save_best_model:
                    self.best_model_state = self.model.state_dict().copy()
            else:
                self.patience_counter += 1

            # Print progress
            if verbose:
                epoch_time = time.time() - epoch_start_time
                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs} ({epoch_time:.2f}s) - "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Train Acc: {train_metrics['accuracy']:.4f}, "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val Acc: {val_metrics['accuracy']:.4f}, "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                )

            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

        # Load best model if saved
        if save_best_model and self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info("Loaded best model state")

        # Save model if path provided
        if model_save_path and save_best_model:
            self.save_model(model_save_path)

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s")

        return self.history

    def _check_improvement(self, current_metric: float) -> bool:
        """Check if current metric improved."""
        if 'loss' in self.early_stopping_metric:
            # Lower is better for loss
            return current_metric < (self.best_metric - self.min_delta)
        else:
            # Higher is better for accuracy/f1/etc
            return current_metric > (self.best_metric + self.min_delta)

    def evaluate(self, test_loader: DataLoader) -> Dict[str, Any]:
        """
        Evaluate the model on test data.

        Args:
            test_loader (DataLoader): Test data loader

        Returns:
            Dict[str, Any]: Evaluation metrics
        """
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_logits = []
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluation"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)

                # Collect results
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_logits.extend(torch.softmax(logits, dim=-1).cpu().numpy())

        # Calculate comprehensive metrics
        avg_loss = total_loss / len(test_loader)
        accuracy = accuracy_score(all_labels, all_predictions)

        # Precision, recall, F1
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_predictions, average=None, zero_division=0
        )

        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='macro', zero_division=0
        )

        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )

        # Confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_predictions)

        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision_per_class': precision.tolist(),
            'recall_per_class': recall.tolist(),
            'f1_per_class': f1.tolist(),
            'support_per_class': support.tolist(),
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
            'confusion_matrix': conf_matrix.tolist(),
            'predictions': all_predictions,
            'true_labels': all_labels,
            'probabilities': all_logits
        }

        return metrics

    def predict(self, data_loader: DataLoader) -> Tuple[List[int], List[int]]:
        """
        Make predictions on data.

        Args:
            data_loader (DataLoader): Data loader

        Returns:
            Tuple[List[int], List[int]]: Predictions and true labels
        """
        self.model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass
                logits = self.model(input_ids, attention_mask)
                predictions = torch.argmax(logits, dim=-1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return all_predictions, all_labels

    def save_model(self, filepath: str, epoch: int = 0, **kwargs):
        """
        Save model checkpoint.

        Args:
            filepath (str): Path to save the model
            epoch (int): Current epoch
            **kwargs: Additional information to save
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': epoch,
            'history': self.history,
            'best_metric': self.best_metric,
            'model_info': self.model.get_model_info() if hasattr(self.model, 'get_model_info') else {},
            **kwargs
        }

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        torch.save(checkpoint, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str, load_optimizer: bool = True, load_scheduler: bool = True):
        """
        Load model checkpoint.

        Args:
            filepath (str): Path to the saved model
            load_optimizer (bool): Whether to load optimizer state
            load_scheduler (bool): Whether to load scheduler state
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if load_scheduler and self.scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler_state = checkpoint['scheduler_state_dict']
            if scheduler_state is not None:
                self.scheduler.load_state_dict(scheduler_state)

        if 'history' in checkpoint:
            self.history = checkpoint['history']

        if 'best_metric' in checkpoint:
            self.best_metric = checkpoint['best_metric']

        logger.info(f"Model loaded from {filepath}")
        logger.info(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")

        return checkpoint

    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get training summary.

        Returns:
            Dict[str, Any]: Training summary
        """
        if not self.history['train_loss']:
            return {"message": "No training history available"}

        summary = {
            'total_epochs': len(self.history['train_loss']),
            'best_train_accuracy': max(self.history['train_accuracy']),
            'best_val_accuracy': max(self.history['val_accuracy']),
            'final_train_loss': self.history['train_loss'][-1],
            'final_val_loss': self.history['val_loss'][-1],
            'final_train_accuracy': self.history['train_accuracy'][-1],
            'final_val_accuracy': self.history['val_accuracy'][-1],
            'best_metric_value': self.best_metric,
            'early_stopping_triggered': self.patience_counter >= self.early_stopping_patience
        }

        return summary