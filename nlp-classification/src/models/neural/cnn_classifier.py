"""
CNN-based text classifier.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from .base import BaseNeuralModel


class CNNClassifier(BaseNeuralModel):
    """
    CNN-based text classifier with multiple filter sizes for n-gram feature extraction.
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int = 100,
        num_filters: int = 100,
        filter_sizes: List[int] = [3, 4, 5],
        dropout: float = 0.5,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False,
        padding_idx: int = 0
    ):
        """
        Initialize CNNClassifier.

        Args:
            vocab_size (int): Size of vocabulary
            num_classes (int): Number of output classes
            embedding_dim (int): Dimension of word embeddings
            num_filters (int): Number of filters for each filter size
            filter_sizes (List[int]): List of filter sizes (kernel sizes)
            dropout (float): Dropout probability
            pretrained_embeddings (torch.Tensor, optional): Pre-trained embeddings
            freeze_embeddings (bool): Whether to freeze embedding weights
            padding_idx (int): Index used for padding tokens
        """
        super(CNNClassifier, self).__init__(
            vocab_size=vocab_size,
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            dropout=dropout,
            pretrained_embeddings=pretrained_embeddings,
            freeze_embeddings=freeze_embeddings,
            padding_idx=padding_idx
        )

        self.num_filters = num_filters
        self.filter_sizes = filter_sizes

        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=filter_size
            )
            for filter_size in filter_sizes
        ])

        # Classification head
        total_filters = num_filters * len(filter_sizes)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(total_filters, total_filters // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(total_filters // 2, num_classes)
        )

        # Initialize weights
        self.initialize_weights()

        print(f"CNN Classifier initialized:")
        print(f"  Num filters: {num_filters}")
        print(f"  Filter sizes: {filter_sizes}")
        print(f"  Total filter outputs: {total_filters}")

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the CNN classifier.

        Args:
            input_ids (torch.Tensor): Input token IDs [batch_size, seq_len]
            attention_mask (torch.Tensor, optional): Attention mask [batch_size, seq_len]

        Returns:
            torch.Tensor: Logits [batch_size, num_classes]
        """
        # Get embeddings
        embeddings = self.get_embeddings(input_ids)  # [batch_size, seq_len, embedding_dim]

        # Apply attention mask if provided
        if attention_mask is not None:
            embeddings = self.apply_attention_mask(embeddings, attention_mask)

        # Transpose for Conv1d: [batch_size, embedding_dim, seq_len]
        embeddings = embeddings.transpose(1, 2)

        # Apply convolutional layers
        conv_outputs = []
        for conv in self.convs:
            # Apply convolution and ReLU activation
            conv_out = F.relu(conv(embeddings))  # [batch_size, num_filters, conv_seq_len]

            # Global max pooling over the sequence dimension
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))  # [batch_size, num_filters, 1]
            pooled = pooled.squeeze(2)  # [batch_size, num_filters]

            conv_outputs.append(pooled)

        # Concatenate all filter outputs
        concatenated = torch.cat(conv_outputs, dim=1)  # [batch_size, total_filters]

        # Apply classifier
        logits = self.classifier(concatenated)

        return logits

    def get_model_info(self):
        """
        Get model-specific information.

        Returns:
            Dict[str, Any]: Model information
        """
        info = super().get_model_info()
        info.update({
            'num_filters': self.num_filters,
            'filter_sizes': self.filter_sizes,
            'total_filters': self.num_filters * len(self.filter_sizes)
        })
        return info


class MultiChannelCNNClassifier(BaseNeuralModel):
    """
    Multi-channel CNN classifier with different embedding channels.
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int = 100,
        num_filters: int = 100,
        filter_sizes: List[int] = [3, 4, 5],
        num_channels: int = 2,
        dropout: float = 0.5,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False,
        padding_idx: int = 0
    ):
        """
        Initialize MultiChannelCNNClassifier.

        Args:
            vocab_size (int): Size of vocabulary
            num_classes (int): Number of output classes
            embedding_dim (int): Dimension of word embeddings
            num_filters (int): Number of filters for each filter size
            filter_sizes (List[int]): List of filter sizes (kernel sizes)
            num_channels (int): Number of embedding channels
            dropout (float): Dropout probability
            pretrained_embeddings (torch.Tensor, optional): Pre-trained embeddings
            freeze_embeddings (bool): Whether to freeze embedding weights
            padding_idx (int): Index used for padding tokens
        """
        super(MultiChannelCNNClassifier, self).__init__(
            vocab_size=vocab_size,
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            dropout=dropout,
            pretrained_embeddings=pretrained_embeddings,
            freeze_embeddings=freeze_embeddings,
            padding_idx=padding_idx
        )

        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.num_channels = num_channels

        # Multiple embedding layers (channels)
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
            for _ in range(num_channels)
        ])

        # Initialize with pre-trained embeddings if provided
        if pretrained_embeddings is not None:
            for embedding in self.embeddings:
                embedding.weight.data.copy_(pretrained_embeddings)

        # Freeze first embedding if requested (static channel)
        if freeze_embeddings and num_channels > 1:
            self.embeddings[0].weight.requires_grad = False

        # Convolutional layers for each channel and filter size
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim * num_channels,
                out_channels=num_filters,
                kernel_size=filter_size
            )
            for filter_size in filter_sizes
        ])

        # Classification head
        total_filters = num_filters * len(filter_sizes)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(total_filters, total_filters // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(total_filters // 2, num_classes)
        )

        # Initialize weights
        self.initialize_weights()

        print(f"Multi-Channel CNN Classifier initialized:")
        print(f"  Num channels: {num_channels}")
        print(f"  Num filters: {num_filters}")
        print(f"  Filter sizes: {filter_sizes}")

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the multi-channel CNN classifier.

        Args:
            input_ids (torch.Tensor): Input token IDs [batch_size, seq_len]
            attention_mask (torch.Tensor, optional): Attention mask [batch_size, seq_len]

        Returns:
            torch.Tensor: Logits [batch_size, num_classes]
        """
        # Get embeddings from all channels
        channel_embeddings = []
        for embedding in self.embeddings:
            emb = embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
            emb = self.dropout_layer(emb)
            channel_embeddings.append(emb)

        # Concatenate channels along the embedding dimension
        embeddings = torch.cat(channel_embeddings, dim=2)  # [batch_size, seq_len, embedding_dim * num_channels]

        # Apply attention mask if provided
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(embeddings).float()
            embeddings = embeddings * mask_expanded

        # Transpose for Conv1d: [batch_size, embedding_dim * num_channels, seq_len]
        embeddings = embeddings.transpose(1, 2)

        # Apply convolutional layers
        conv_outputs = []
        for conv in self.convs:
            # Apply convolution and ReLU activation
            conv_out = F.relu(conv(embeddings))  # [batch_size, num_filters, conv_seq_len]

            # Global max pooling over the sequence dimension
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))  # [batch_size, num_filters, 1]
            pooled = pooled.squeeze(2)  # [batch_size, num_filters]

            conv_outputs.append(pooled)

        # Concatenate all filter outputs
        concatenated = torch.cat(conv_outputs, dim=1)  # [batch_size, total_filters]

        # Apply classifier
        logits = self.classifier(concatenated)

        return logits

    def get_model_info(self):
        """
        Get model-specific information.

        Returns:
            Dict[str, Any]: Model information
        """
        info = super().get_model_info()
        info.update({
            'num_channels': self.num_channels,
            'num_filters': self.num_filters,
            'filter_sizes': self.filter_sizes,
            'total_filters': self.num_filters * len(self.filter_sizes)
        })
        return info


class HierarchicalCNNClassifier(BaseNeuralModel):
    """
    Hierarchical CNN classifier with multiple convolutional layers.
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int = 100,
        filter_configs: List[dict] = None,
        dropout: float = 0.5,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False,
        padding_idx: int = 0
    ):
        """
        Initialize HierarchicalCNNClassifier.

        Args:
            vocab_size (int): Size of vocabulary
            num_classes (int): Number of output classes
            embedding_dim (int): Dimension of word embeddings
            filter_configs (List[dict]): List of filter configurations
            dropout (float): Dropout probability
            pretrained_embeddings (torch.Tensor, optional): Pre-trained embeddings
            freeze_embeddings (bool): Whether to freeze embedding weights
            padding_idx (int): Index used for padding tokens
        """
        super(HierarchicalCNNClassifier, self).__init__(
            vocab_size=vocab_size,
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            dropout=dropout,
            pretrained_embeddings=pretrained_embeddings,
            freeze_embeddings=freeze_embeddings,
            padding_idx=padding_idx
        )

        # Default filter configurations if not provided
        if filter_configs is None:
            filter_configs = [
                {'num_filters': 100, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'num_filters': 100, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'num_filters': 100, 'kernel_size': 3, 'stride': 1, 'padding': 1}
            ]

        self.filter_configs = filter_configs

        # Build convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = embedding_dim

        for i, config in enumerate(filter_configs):
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=config['num_filters'],
                kernel_size=config['kernel_size'],
                stride=config.get('stride', 1),
                padding=config.get('padding', 0)
            )
            self.conv_layers.append(conv)
            in_channels = config['num_filters']

        # Global pooling and classification
        final_filters = filter_configs[-1]['num_filters']
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(final_filters, final_filters // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_filters // 2, num_classes)
        )

        # Initialize weights
        self.initialize_weights()

        print(f"Hierarchical CNN Classifier initialized:")
        print(f"  Num layers: {len(filter_configs)}")
        print(f"  Filter configs: {filter_configs}")

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the hierarchical CNN classifier.

        Args:
            input_ids (torch.Tensor): Input token IDs [batch_size, seq_len]
            attention_mask (torch.Tensor, optional): Attention mask [batch_size, seq_len]

        Returns:
            torch.Tensor: Logits [batch_size, num_classes]
        """
        # Get embeddings
        x = self.get_embeddings(input_ids)  # [batch_size, seq_len, embedding_dim]

        # Apply attention mask if provided
        if attention_mask is not None:
            x = self.apply_attention_mask(x, attention_mask)

        # Transpose for Conv1d: [batch_size, embedding_dim, seq_len]
        x = x.transpose(1, 2)

        # Pass through convolutional layers
        for i, conv in enumerate(self.conv_layers):
            x = F.relu(conv(x))  # [batch_size, num_filters, seq_len]

            # Apply dropout between layers
            if i < len(self.conv_layers) - 1:
                x = self.dropout_layer(x)

        # Global max pooling
        x = F.max_pool1d(x, kernel_size=x.size(2))  # [batch_size, num_filters, 1]
        x = x.squeeze(2)  # [batch_size, num_filters]

        # Apply classifier
        logits = self.classifier(x)

        return logits

    def get_model_info(self):
        """
        Get model-specific information.

        Returns:
            Dict[str, Any]: Model information
        """
        info = super().get_model_info()
        info.update({
            'num_conv_layers': len(self.filter_configs),
            'filter_configs': self.filter_configs
        })
        return info