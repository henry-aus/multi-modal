"""
Custom dataset loader for image classification
Handles loading images from the data format: category_id\\subfolder\\filename.jpg
"""

import os
from pathlib import Path
from typing import Tuple, List, Optional, Callable

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class ImageClassificationDataset(Dataset):
    """Custom dataset for image classification with category-based folder structure"""

    def __init__(
        self,
        data_file: str,
        data_root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        """
        Initialize the dataset

        Args:
            data_file: Path to the txt file containing image paths and labels
            data_root: Root directory containing the images
            transform: Transform to apply to images
            target_transform: Transform to apply to labels
        """
        self.data_root = Path(data_root)
        self.transform = transform
        self.target_transform = target_transform

        # Load image paths and labels
        self.samples = self._load_samples(data_file)

        # Get unique classes and create label mapping
        self.classes = sorted(list(set(label for _, label in self.samples)))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

    def _load_samples(self, data_file: str) -> List[Tuple[str, int]]:
        """Load samples from the data file"""
        samples = []

        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Parse the line: category_id\subfolder\filename.jpg
                # Extract category_id from the path
                parts = line.split('\\')
                if len(parts) < 2:
                    continue

                category_id = int(parts[0])
                image_path = os.path.join(self.data_root, line.replace('\\', os.sep))

                samples.append((image_path, category_id))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample from the dataset"""
        image_path, label = self.samples[idx]

        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except (IOError, OSError) as e:
            print(f"Warning: Could not load image {image_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color='black')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Convert label to class index (0-based)
        label_idx = self.class_to_idx[label]
        if self.target_transform:
            label_idx = self.target_transform(label_idx)

        return image, label_idx

    def get_class_counts(self) -> dict:
        """Get the count of samples per class"""
        class_counts = {}
        for _, label in self.samples:
            class_counts[label] = class_counts.get(label, 0) + 1
        return class_counts


def get_transforms(split: str = 'train', image_size: int = 224) -> transforms.Compose:
    """
    Get data transforms for training and validation

    Args:
        split: 'train' or 'val'
        image_size: Target image size

    Returns:
        Composed transforms
    """
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((int(image_size * 1.12), int(image_size * 1.12))),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def create_data_loaders(
    train_file: str,
    test_file: str,
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and test data loaders

    Args:
        train_file: Path to training data file
        test_file: Path to test data file
        data_root: Root directory containing images
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        image_size: Target image size

    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Create transforms
    train_transform = get_transforms('train', image_size)
    test_transform = get_transforms('val', image_size)

    # Create datasets
    train_dataset = ImageClassificationDataset(
        train_file, data_root, transform=train_transform
    )
    test_dataset = ImageClassificationDataset(
        test_file, data_root, transform=test_transform
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader