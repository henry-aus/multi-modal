"""
PyTorch dataset classes for NLP classification.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Tuple, Optional, Union
import pandas as pd
import numpy as np
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class Vocabulary:
    """
    Vocabulary management for text data.
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        min_frequency: int = 2,
        pad_token: str = "<PAD>",
        unk_token: str = "<UNK>",
        sos_token: str = "<SOS>",
        eos_token: str = "<EOS>"
    ):
        """
        Initialize Vocabulary.

        Args:
            vocab_size (int): Maximum vocabulary size
            min_frequency (int): Minimum frequency for a word to be included
            pad_token (str): Padding token
            unk_token (str): Unknown token
            sos_token (str): Start of sequence token
            eos_token (str): End of sequence token
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.sos_token = sos_token
        self.eos_token = eos_token

        # Special tokens
        self.special_tokens = [pad_token, unk_token, sos_token, eos_token]

        # Mappings
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_frequencies = Counter()

        # Initialize with special tokens
        for i, token in enumerate(self.special_tokens):
            self.word_to_idx[token] = i
            self.idx_to_word[i] = token

        self.current_idx = len(self.special_tokens)

    def build_from_texts(self, texts: List[str]) -> 'Vocabulary':
        """
        Build vocabulary from texts.

        Args:
            texts (List[str]): List of texts

        Returns:
            Vocabulary: Self
        """
        # Count word frequencies
        for text in texts:
            words = text.split()
            self.word_frequencies.update(words)

        # Get most common words
        most_common = self.word_frequencies.most_common(
            self.vocab_size - len(self.special_tokens)
        )

        # Add words that meet minimum frequency requirement
        for word, freq in most_common:
            if freq >= self.min_frequency and word not in self.word_to_idx:
                self.word_to_idx[word] = self.current_idx
                self.idx_to_word[self.current_idx] = word
                self.current_idx += 1

        logger.info(f"Built vocabulary with {len(self.word_to_idx)} words")
        logger.info(f"Most frequent words: {list(dict(most_common[:10]).keys())}")

        return self

    def text_to_indices(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """
        Convert text to list of indices.

        Args:
            text (str): Input text
            max_length (int, optional): Maximum sequence length

        Returns:
            List[int]: List of word indices
        """
        words = text.split()
        indices = []

        for word in words:
            if word in self.word_to_idx:
                indices.append(self.word_to_idx[word])
            else:
                indices.append(self.word_to_idx[self.unk_token])

        # Truncate or pad if max_length is specified
        if max_length is not None:
            if len(indices) > max_length:
                indices = indices[:max_length]
            else:
                # Pad with PAD token
                pad_length = max_length - len(indices)
                indices.extend([self.word_to_idx[self.pad_token]] * pad_length)

        return indices

    def indices_to_text(self, indices: List[int]) -> str:
        """
        Convert list of indices back to text.

        Args:
            indices (List[int]): List of word indices

        Returns:
            str: Reconstructed text
        """
        words = []
        for idx in indices:
            if idx in self.idx_to_word:
                word = self.idx_to_word[idx]
                # Skip special tokens when reconstructing
                if word not in [self.pad_token, self.sos_token, self.eos_token]:
                    words.append(word)
        return ' '.join(words)

    def __len__(self) -> int:
        """Get vocabulary size."""
        return len(self.word_to_idx)

    def get_pad_idx(self) -> int:
        """Get padding token index."""
        return self.word_to_idx[self.pad_token]

    def get_unk_idx(self) -> int:
        """Get unknown token index."""
        return self.word_to_idx[self.unk_token]


class TextClassificationDataset(Dataset):
    """
    PyTorch Dataset for text classification.
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        vocabulary: Vocabulary,
        max_length: int = 512,
        return_attention_mask: bool = True
    ):
        """
        Initialize TextClassificationDataset.

        Args:
            texts (List[str]): List of texts
            labels (List[int]): List of encoded labels
            vocabulary (Vocabulary): Vocabulary object
            max_length (int): Maximum sequence length
            return_attention_mask (bool): Whether to return attention masks
        """
        self.texts = texts
        self.labels = labels
        self.vocabulary = vocabulary
        self.max_length = max_length
        self.return_attention_mask = return_attention_mask

        if len(texts) != len(labels):
            raise ValueError("Number of texts and labels must match")

        logger.info(f"Created dataset with {len(self)} samples")

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset.

        Args:
            idx (int): Item index

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing input tensors
        """
        text = self.texts[idx]
        label = self.labels[idx]

        # Convert text to indices
        indices = self.vocabulary.text_to_indices(text, self.max_length)

        # Create tensors
        input_ids = torch.tensor(indices, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)

        result = {
            'input_ids': input_ids,
            'labels': label_tensor
        }

        # Create attention mask if requested
        if self.return_attention_mask:
            # Attention mask: 1 for real tokens, 0 for padding
            attention_mask = (input_ids != self.vocabulary.get_pad_idx()).long()
            result['attention_mask'] = attention_mask

        return result

    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for imbalanced datasets.

        Returns:
            torch.Tensor: Class weights
        """
        label_counts = Counter(self.labels)
        num_classes = len(label_counts)
        total_samples = len(self.labels)

        # Calculate weights: inverse frequency
        weights = []
        for class_id in range(num_classes):
            if class_id in label_counts:
                weight = total_samples / (num_classes * label_counts[class_id])
            else:
                weight = 1.0
            weights.append(weight)

        return torch.tensor(weights, dtype=torch.float)

    def get_label_distribution(self) -> Dict[int, int]:
        """
        Get label distribution.

        Returns:
            Dict[int, int]: Label counts
        """
        return dict(Counter(self.labels))


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching.

    Args:
        batch (List[Dict[str, torch.Tensor]]): List of samples

    Returns:
        Dict[str, torch.Tensor]: Batched tensors
    """
    # Stack tensors
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])

    result = {
        'input_ids': input_ids,
        'labels': labels
    }

    # Add attention mask if present
    if 'attention_mask' in batch[0]:
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        result['attention_mask'] = attention_mask

    return result


def create_data_loaders(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    vocabulary: Vocabulary,
    batch_size: int = 32,
    max_length: int = 512,
    num_workers: int = 0,
    shuffle_train: bool = True,
    return_attention_mask: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders.

    Args:
        train_texts (List[str]): Training texts
        train_labels (List[int]): Training labels
        val_texts (List[str]): Validation texts
        val_labels (List[int]): Validation labels
        vocabulary (Vocabulary): Vocabulary object
        batch_size (int): Batch size
        max_length (int): Maximum sequence length
        num_workers (int): Number of workers for data loading
        shuffle_train (bool): Whether to shuffle training data
        return_attention_mask (bool): Whether to return attention masks

    Returns:
        Tuple[DataLoader, DataLoader]: Train and validation data loaders
    """
    # Create datasets
    train_dataset = TextClassificationDataset(
        train_texts, train_labels, vocabulary, max_length, return_attention_mask
    )

    val_dataset = TextClassificationDataset(
        val_texts, val_labels, vocabulary, max_length, return_attention_mask
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )

    logger.info(f"Created data loaders:")
    logger.info(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    logger.info(f"  Validation: {len(val_dataset)} samples, {len(val_loader)} batches")

    return train_loader, val_loader


def get_embeddings_matrix(
    vocabulary: Vocabulary,
    embedding_path: str,
    embedding_dim: int = 100
) -> torch.Tensor:
    """
    Load pre-trained embeddings and create embedding matrix.

    Args:
        vocabulary (Vocabulary): Vocabulary object
        embedding_path (str): Path to pre-trained embeddings file
        embedding_dim (int): Embedding dimension

    Returns:
        torch.Tensor: Embedding matrix
    """
    embeddings_matrix = torch.randn(len(vocabulary), embedding_dim)

    # Initialize special tokens with zeros
    pad_idx = vocabulary.get_pad_idx()
    embeddings_matrix[pad_idx] = torch.zeros(embedding_dim)

    found_words = 0

    try:
        with open(embedding_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != embedding_dim + 1:
                    continue

                word = parts[0]
                if word in vocabulary.word_to_idx:
                    idx = vocabulary.word_to_idx[word]
                    embedding = torch.tensor([float(x) for x in parts[1:]])
                    embeddings_matrix[idx] = embedding
                    found_words += 1

        logger.info(f"Loaded embeddings for {found_words}/{len(vocabulary)} words")

    except Exception as e:
        logger.error(f"Error loading embeddings from {embedding_path}: {e}")
        logger.info("Using random embeddings")

    return embeddings_matrix


if __name__ == "__main__":
    # Test dataset functionality
    logging.basicConfig(level=logging.INFO)

    print("Dataset Test")
    print("=" * 50)

    # Sample data
    texts = [
        "this is a positive example",
        "this is a negative example",
        "another positive case here",
        "negative example again",
        "final positive sample"
    ]

    labels = [1, 0, 1, 0, 1]  # Binary classification

    # Test vocabulary
    print("Testing Vocabulary:")
    vocab = Vocabulary(vocab_size=100, min_frequency=1)
    vocab.build_from_texts(texts)

    print(f"Vocabulary size: {len(vocab)}")
    print(f"Sample text to indices: {vocab.text_to_indices(texts[0])}")
    print(f"Indices back to text: {vocab.indices_to_text(vocab.text_to_indices(texts[0]))}")

    # Test dataset
    print("\nTesting Dataset:")
    dataset = TextClassificationDataset(texts, labels, vocab, max_length=10)

    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Input shape: {sample['input_ids'].shape}")
    print(f"Label: {sample['labels'].item()}")

    if 'attention_mask' in sample:
        print(f"Attention mask: {sample['attention_mask']}")

    # Test data loader
    print("\nTesting DataLoader:")
    train_loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    for batch in train_loader:
        print(f"Batch input shape: {batch['input_ids'].shape}")
        print(f"Batch labels shape: {batch['labels'].shape}")
        if 'attention_mask' in batch:
            print(f"Batch attention mask shape: {batch['attention_mask'].shape}")
        break

    # Test class weights
    print("\nTesting Class Weights:")
    class_weights = dataset.get_class_weights()
    print(f"Class weights: {class_weights}")

    label_dist = dataset.get_label_distribution()
    print(f"Label distribution: {label_dist}")