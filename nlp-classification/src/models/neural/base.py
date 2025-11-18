"""
Base class for PyTorch neural network models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class BaseNeuralModel(nn.Module, ABC):
    """
    Base class for PyTorch neural network text classification models.
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int = 100,
        dropout: float = 0.3,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False,
        padding_idx: int = 0
    ):
        """
        Initialize BaseNeuralModel.

        Args:
            vocab_size (int): Size of vocabulary
            num_classes (int): Number of output classes
            embedding_dim (int): Dimension of word embeddings
            dropout (float): Dropout probability
            pretrained_embeddings (torch.Tensor, optional): Pre-trained embeddings
            freeze_embeddings (bool): Whether to freeze embedding weights
            padding_idx (int): Index used for padding tokens
        """
        super(BaseNeuralModel, self).__init__()

        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.freeze_embeddings = freeze_embeddings
        self.padding_idx = padding_idx

        # Create embedding layer
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )

        # Initialize with pre-trained embeddings if provided
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            logger.info("Initialized with pre-trained embeddings")

        # Freeze embeddings if requested
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False
            logger.info("Frozen embedding weights")

        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)

        logger.info(f"Initialized {self.__class__.__name__}")
        logger.info(f"  Vocab size: {vocab_size}")
        logger.info(f"  Num classes: {num_classes}")
        logger.info(f"  Embedding dim: {embedding_dim}")
        logger.info(f"  Dropout: {dropout}")

    @abstractmethod
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs [batch_size, seq_len]
            attention_mask (torch.Tensor, optional): Attention mask [batch_size, seq_len]

        Returns:
            torch.Tensor: Logits [batch_size, num_classes]
        """
        pass

    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get word embeddings for input tokens.

        Args:
            input_ids (torch.Tensor): Input token IDs [batch_size, seq_len]

        Returns:
            torch.Tensor: Embeddings [batch_size, seq_len, embedding_dim]
        """
        embeddings = self.embedding(input_ids)
        embeddings = self.dropout_layer(embeddings)
        return embeddings

    def apply_attention_mask(self, embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply attention mask to embeddings by setting padded positions to zero.

        Args:
            embeddings (torch.Tensor): Input embeddings [batch_size, seq_len, embedding_dim]
            attention_mask (torch.Tensor): Attention mask [batch_size, seq_len]

        Returns:
            torch.Tensor: Masked embeddings [batch_size, seq_len, embedding_dim]
        """
        if attention_mask is not None:
            # Expand attention mask to match embedding dimensions
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(embeddings).float()
            embeddings = embeddings * mask_expanded

        return embeddings

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor, class_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute classification loss.

        Args:
            logits (torch.Tensor): Model predictions [batch_size, num_classes]
            labels (torch.Tensor): True labels [batch_size]
            class_weights (torch.Tensor, optional): Class weights for imbalanced data

        Returns:
            torch.Tensor: Loss value
        """
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        loss = criterion(logits, labels)
        return loss

    def predict(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Make predictions (returns class indices).

        Args:
            input_ids (torch.Tensor): Input token IDs [batch_size, seq_len]
            attention_mask (torch.Tensor, optional): Attention mask [batch_size, seq_len]

        Returns:
            torch.Tensor: Predicted class indices [batch_size]
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=-1)
        return predictions

    def predict_proba(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get prediction probabilities.

        Args:
            input_ids (torch.Tensor): Input token IDs [batch_size, seq_len]
            attention_mask (torch.Tensor, optional): Attention mask [batch_size, seq_len]

        Returns:
            torch.Tensor: Prediction probabilities [batch_size, num_classes]
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            probabilities = F.softmax(logits, dim=-1)
        return probabilities

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and statistics.

        Returns:
            Dict[str, Any]: Model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        info = {
            'model_type': self.__class__.__name__,
            'vocab_size': self.vocab_size,
            'num_classes': self.num_classes,
            'embedding_dim': self.embedding_dim,
            'dropout': self.dropout,
            'freeze_embeddings': self.freeze_embeddings,
            'padding_idx': self.padding_idx,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params
        }

        return info

    def save_model(self, filepath: str, optimizer: Optional[torch.optim.Optimizer] = None, epoch: int = 0, loss: float = 0.0):
        """
        Save model state to file.

        Args:
            filepath (str): Path to save the model
            optimizer (torch.optim.Optimizer, optional): Optimizer state to save
            epoch (int): Current epoch number
            loss (float): Current loss value
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_info': self.get_model_info(),
            'epoch': epoch,
            'loss': loss
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        torch.save(checkpoint, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str, optimizer: Optional[torch.optim.Optimizer] = None, device: torch.device = torch.device('cpu')):
        """
        Load model state from file.

        Args:
            filepath (str): Path to the saved model
            optimizer (torch.optim.Optimizer, optional): Optimizer to load state into
            device (torch.device): Device to load model on

        Returns:
            Dict[str, Any]: Checkpoint information
        """
        checkpoint = torch.load(filepath, map_location=device)

        self.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        logger.info(f"Model loaded from {filepath}")
        logger.info(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        logger.info(f"  Loss: {checkpoint.get('loss', 'N/A')}")

        return checkpoint

    def initialize_weights(self):
        """
        Initialize model weights using best practices.
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization for linear layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.LSTM):
                # Initialize LSTM weights
                for param_name, param in module.named_parameters():
                    if 'weight_ih' in param_name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in param_name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in param_name:
                        param.data.fill_(0)
                        # Set forget gate bias to 1
                        n = param.size(0)
                        param.data[n//4:n//2].fill_(1)

            elif isinstance(module, nn.Conv1d):
                # Initialize CNN weights
                nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.Embedding):
                # Don't reinitialize if using pre-trained embeddings
                if not self.freeze_embeddings:
                    nn.init.uniform_(module.weight, -0.1, 0.1)
                    if module.padding_idx is not None:
                        module.weight.data[module.padding_idx].fill_(0)

        logger.info("Model weights initialized")


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in a model.

    Args:
        model (nn.Module): PyTorch model

    Returns:
        Tuple[int, int]: (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def freeze_layers(model: nn.Module, layer_names: list):
    """
    Freeze specified layers in the model.

    Args:
        model (nn.Module): PyTorch model
        layer_names (list): List of layer names to freeze
    """
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if layer_name in name:
                param.requires_grad = False
                logger.info(f"Frozen layer: {name}")


def unfreeze_layers(model: nn.Module, layer_names: list):
    """
    Unfreeze specified layers in the model.

    Args:
        model (nn.Module): PyTorch model
        layer_names (list): List of layer names to unfreeze
    """
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if layer_name in name:
                param.requires_grad = True
                logger.info(f"Unfrozen layer: {name}")