"""
LSTM-based text classifier.
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
from .base import BaseNeuralModel


class LSTMClassifier(BaseNeuralModel):
    """
    LSTM-based text classifier with optional bidirectional processing.
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int = 100,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False,
        padding_idx: int = 0
    ):
        """
        Initialize LSTMClassifier.

        Args:
            vocab_size (int): Size of vocabulary
            num_classes (int): Number of output classes
            embedding_dim (int): Dimension of word embeddings
            hidden_dim (int): Hidden dimension of LSTM
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout probability
            bidirectional (bool): Whether to use bidirectional LSTM
            pretrained_embeddings (torch.Tensor, optional): Pre-trained embeddings
            freeze_embeddings (bool): Whether to freeze embedding weights
            padding_idx (int): Index used for padding tokens
        """
        super(LSTMClassifier, self).__init__(
            vocab_size=vocab_size,
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            dropout=dropout,
            pretrained_embeddings=pretrained_embeddings,
            freeze_embeddings=freeze_embeddings,
            padding_idx=padding_idx
        )

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        # Calculate the size of LSTM output
        lstm_output_size = hidden_dim * 2 if bidirectional else hidden_dim

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 2, num_classes)
        )

        # Initialize weights
        self.initialize_weights()

        print(f"LSTM Classifier initialized:")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Num layers: {num_layers}")
        print(f"  Bidirectional: {bidirectional}")
        print(f"  LSTM output size: {lstm_output_size}")

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden state for LSTM.

        Args:
            batch_size (int): Batch size
            device (torch.device): Device to create tensors on

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Initial hidden and cell states
        """
        num_directions = 2 if self.bidirectional else 1

        h0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_dim,
            device=device
        )

        c0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_dim,
            device=device
        )

        return h0, c0

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the LSTM classifier.

        Args:
            input_ids (torch.Tensor): Input token IDs [batch_size, seq_len]
            attention_mask (torch.Tensor, optional): Attention mask [batch_size, seq_len]

        Returns:
            torch.Tensor: Logits [batch_size, num_classes]
        """
        batch_size, seq_len = input_ids.size()

        # Get embeddings
        embeddings = self.get_embeddings(input_ids)  # [batch_size, seq_len, embedding_dim]

        # Apply attention mask if provided
        if attention_mask is not None:
            embeddings = self.apply_attention_mask(embeddings, attention_mask)

        # Initialize hidden state
        device = input_ids.device
        h0, c0 = self.init_hidden(batch_size, device)

        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(embeddings, (h0, c0))
        # lstm_out: [batch_size, seq_len, hidden_dim * num_directions]

        # Use the final hidden state for classification
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            # hn: [num_layers * num_directions, batch_size, hidden_dim]
            forward_final = hn[-2, :, :]  # Last layer, forward direction
            backward_final = hn[-1, :, :] # Last layer, backward direction
            final_hidden = torch.cat([forward_final, backward_final], dim=1)
        else:
            final_hidden = hn[-1, :, :]  # Last layer

        # Apply classifier
        logits = self.classifier(final_hidden)  # [batch_size, num_classes]

        return logits

    def forward_with_attention_pooling(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Alternative forward pass using attention-weighted pooling instead of final hidden state.

        Args:
            input_ids (torch.Tensor): Input token IDs [batch_size, seq_len]
            attention_mask (torch.Tensor, optional): Attention mask [batch_size, seq_len]

        Returns:
            torch.Tensor: Logits [batch_size, num_classes]
        """
        batch_size, seq_len = input_ids.size()

        # Get embeddings
        embeddings = self.get_embeddings(input_ids)

        # Apply attention mask if provided
        if attention_mask is not None:
            embeddings = self.apply_attention_mask(embeddings, attention_mask)

        # Initialize hidden state
        device = input_ids.device
        h0, c0 = self.init_hidden(batch_size, device)

        # LSTM forward pass
        lstm_out, _ = self.lstm(embeddings, (h0, c0))
        # lstm_out: [batch_size, seq_len, hidden_dim * num_directions]

        # Attention-weighted pooling
        if attention_mask is not None:
            # Create attention weights (simple approach: uniform over non-padded tokens)
            attention_weights = attention_mask.float() / attention_mask.sum(dim=1, keepdim=True)
            attention_weights = attention_weights.unsqueeze(-1)  # [batch_size, seq_len, 1]

            # Apply attention weights
            pooled_output = (lstm_out * attention_weights).sum(dim=1)  # [batch_size, hidden_dim * num_directions]
        else:
            # Simple mean pooling
            pooled_output = lstm_out.mean(dim=1)

        # Apply classifier
        logits = self.classifier(pooled_output)

        return logits

    def get_model_info(self):
        """
        Get model-specific information.

        Returns:
            Dict[str, Any]: Model information
        """
        info = super().get_model_info()
        info.update({
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'bidirectional': self.bidirectional,
            'lstm_output_size': self.hidden_dim * (2 if self.bidirectional else 1)
        })
        return info


class BiLSTMClassifier(LSTMClassifier):
    """
    Bidirectional LSTM classifier (convenience class).
    """

    def __init__(self, **kwargs):
        """
        Initialize BiLSTMClassifier.

        Args:
            **kwargs: Arguments for LSTMClassifier
        """
        kwargs['bidirectional'] = True
        super(BiLSTMClassifier, self).__init__(**kwargs)


class StackedLSTMClassifier(BaseNeuralModel):
    """
    Stacked LSTM classifier with residual connections.
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int = 100,
        hidden_dims: list = [128, 128, 64],
        dropout: float = 0.3,
        bidirectional: bool = True,
        use_residual: bool = True,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False,
        padding_idx: int = 0
    ):
        """
        Initialize StackedLSTMClassifier.

        Args:
            vocab_size (int): Size of vocabulary
            num_classes (int): Number of output classes
            embedding_dim (int): Dimension of word embeddings
            hidden_dims (list): List of hidden dimensions for each LSTM layer
            dropout (float): Dropout probability
            bidirectional (bool): Whether to use bidirectional LSTM
            use_residual (bool): Whether to use residual connections
            pretrained_embeddings (torch.Tensor, optional): Pre-trained embeddings
            freeze_embeddings (bool): Whether to freeze embedding weights
            padding_idx (int): Index used for padding tokens
        """
        super(StackedLSTMClassifier, self).__init__(
            vocab_size=vocab_size,
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            dropout=dropout,
            pretrained_embeddings=pretrained_embeddings,
            freeze_embeddings=freeze_embeddings,
            padding_idx=padding_idx
        )

        self.hidden_dims = hidden_dims
        self.bidirectional = bidirectional
        self.use_residual = use_residual
        self.num_layers = len(hidden_dims)

        # Create stacked LSTM layers
        self.lstm_layers = nn.ModuleList()
        input_size = embedding_dim

        for i, hidden_dim in enumerate(hidden_dims):
            lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_dim,
                num_layers=1,
                dropout=0,  # We'll handle dropout manually
                bidirectional=bidirectional,
                batch_first=True
            )
            self.lstm_layers.append(lstm)

            # Update input size for next layer
            input_size = hidden_dim * (2 if bidirectional else 1)

        # Projection layers for residual connections
        if use_residual:
            self.projection_layers = nn.ModuleList()
            input_size = embedding_dim
            for hidden_dim in hidden_dims:
                output_size = hidden_dim * (2 if bidirectional else 1)
                if input_size != output_size:
                    proj = nn.Linear(input_size, output_size)
                else:
                    proj = nn.Identity()
                self.projection_layers.append(proj)
                input_size = output_size

        # Classification head
        final_hidden_size = hidden_dims[-1] * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(final_hidden_size, final_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_hidden_size // 2, num_classes)
        )

        # Initialize weights
        self.initialize_weights()

        print(f"Stacked LSTM Classifier initialized:")
        print(f"  Hidden dims: {hidden_dims}")
        print(f"  Bidirectional: {bidirectional}")
        print(f"  Residual connections: {use_residual}")

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the stacked LSTM classifier.

        Args:
            input_ids (torch.Tensor): Input token IDs [batch_size, seq_len]
            attention_mask (torch.Tensor, optional): Attention mask [batch_size, seq_len]

        Returns:
            torch.Tensor: Logits [batch_size, num_classes]
        """
        # Get embeddings
        x = self.get_embeddings(input_ids)

        # Apply attention mask if provided
        if attention_mask is not None:
            x = self.apply_attention_mask(x, attention_mask)

        # Pass through stacked LSTM layers
        for i, lstm in enumerate(self.lstm_layers):
            lstm_out, _ = lstm(x)

            # Apply residual connection if enabled
            if self.use_residual and hasattr(self, 'projection_layers'):
                residual = self.projection_layers[i](x)
                lstm_out = lstm_out + residual

            # Apply dropout
            lstm_out = self.dropout_layer(lstm_out)
            x = lstm_out

        # Use mean pooling for final representation
        if attention_mask is not None:
            # Masked mean pooling
            attention_weights = attention_mask.float() / attention_mask.sum(dim=1, keepdim=True)
            attention_weights = attention_weights.unsqueeze(-1)
            pooled_output = (x * attention_weights).sum(dim=1)
        else:
            pooled_output = x.mean(dim=1)

        # Apply classifier
        logits = self.classifier(pooled_output)

        return logits

    def get_model_info(self):
        """
        Get model-specific information.

        Returns:
            Dict[str, Any]: Model information
        """
        info = super().get_model_info()
        info.update({
            'hidden_dims': self.hidden_dims,
            'bidirectional': self.bidirectional,
            'use_residual': self.use_residual,
            'num_lstm_layers': self.num_layers
        })
        return info