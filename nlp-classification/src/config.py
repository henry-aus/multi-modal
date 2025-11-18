"""
Configuration management system for the NLP classification project.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Data processing configuration."""

    max_sequence_length: int = 512
    vocab_size: int = 10000
    min_word_frequency: int = 2
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    stratify: bool = True
    text_column: str = "text"
    label_column: str = "label"
    lowercase: bool = True
    remove_punctuation: bool = True
    remove_stopwords: bool = True
    stemming: bool = False
    lemmatization: bool = True


@dataclass
class ModelConfig:
    """Base model configuration."""

    model_type: str = "lstm"  # lstm, cnn, naive_bayes, svm, logistic_regression
    random_state: int = 42


@dataclass
class LSTMConfig(ModelConfig):
    """LSTM model configuration."""

    model_type: str = "lstm"
    embedding_dim: int = 100
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True
    freeze_embeddings: bool = False
    pretrained_embeddings: Optional[str] = None  # Path to embeddings file


@dataclass
class CNNConfig(ModelConfig):
    """CNN model configuration."""

    model_type: str = "cnn"
    embedding_dim: int = 100
    num_filters: int = 100
    filter_sizes: list = field(default_factory=lambda: [3, 4, 5])
    dropout: float = 0.5
    freeze_embeddings: bool = False
    pretrained_embeddings: Optional[str] = None


@dataclass
class ClassicalMLConfig(ModelConfig):
    """Classical ML model configuration."""

    vectorizer: str = "tfidf"  # tfidf, count
    max_features: int = 10000
    ngram_range: tuple = field(default_factory=lambda: (1, 2))

    # Naive Bayes specific
    alpha: float = 1.0

    # SVM specific
    svm_kernel: str = "rbf"
    svm_C: float = 1.0
    svm_gamma: str = "scale"

    # Logistic Regression specific
    lr_C: float = 1.0
    lr_penalty: str = "l2"
    lr_solver: str = "lbfgs"
    lr_max_iter: int = 1000


@dataclass
class TrainingConfig:
    """Training configuration."""

    batch_size: int = 64
    learning_rate: float = 0.001
    num_epochs: int = 200
    early_stopping_patience: int = 10
    early_stopping_metric: str = "val_loss"  # val_loss, val_accuracy
    min_delta: float = 0.001
    optimizer: str = "adam"  # adam, sgd, rmsprop
    scheduler: str = "step"  # step, cosine, plateau
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    gradient_clip_norm: float = 1.0
    weight_decay: float = 0.0
    warmup_steps: int = 0


@dataclass
class CrossValidationConfig:
    """Cross-validation configuration."""

    enabled: bool = True
    n_splits: int = 5
    shuffle: bool = True
    stratify: bool = True
    random_state: int = 42
    target_accuracy: float = 0.8  # Required minimum accuracy


@dataclass
class LoggingConfig:
    """Logging and monitoring configuration."""

    log_level: str = "INFO"
    log_dir: str = "outputs/logs"
    tensorboard_dir: str = "outputs/tensorboard"
    save_model_every_n_epochs: int = 10
    log_metrics_every_n_steps: int = 100
    save_best_model: bool = True
    save_last_model: bool = True


@dataclass
class Config:
    """Main configuration class."""

    project_name: str = "nlp-classification"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    cross_validation: CrossValidationConfig = field(
        default_factory=CrossValidationConfig
    )
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Paths
    data_dir: str = "data"
    output_dir: str = "outputs"
    model_dir: str = "outputs/models"

    # Device
    device: str = "auto"  # auto, cpu, cuda, mps

    # Reproducibility
    seed: int = 42


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path (str): Path to the YAML configuration file

    Returns:
        Config: Loaded configuration object
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return create_config_from_dict(config_dict)


def create_config_from_dict(config_dict: Dict[str, Any]) -> Config:
    """
    Create configuration object from dictionary.

    Args:
        config_dict (Dict[str, Any]): Configuration dictionary

    Returns:
        Config: Configuration object
    """
    # Extract model-specific configuration
    model_type = config_dict.get("model", {}).get("model_type", "lstm")

    # Create appropriate model config
    if model_type == "lstm":
        model_config = LSTMConfig(**config_dict.get("model", {}))
    elif model_type == "cnn":
        model_config = CNNConfig(**config_dict.get("model", {}))
    elif model_type in ["naive_bayes", "svm", "logistic_regression"]:
        model_config = ClassicalMLConfig(**config_dict.get("model", {}))
    else:
        model_config = ModelConfig(**config_dict.get("model", {}))

    # Create other configs
    data_config = DataConfig(**config_dict.get("data", {}))
    training_config = TrainingConfig(**config_dict.get("training", {}))
    cv_config = CrossValidationConfig(**config_dict.get("cross_validation", {}))
    logging_config = LoggingConfig(**config_dict.get("logging", {}))

    # Create main config
    config = Config(
        project_name=config_dict.get("project_name", "nlp-classification"),
        data=data_config,
        model=model_config,
        training=training_config,
        cross_validation=cv_config,
        logging=logging_config,
        data_dir=config_dict.get("data_dir", "data"),
        output_dir=config_dict.get("output_dir", "outputs"),
        model_dir=config_dict.get("model_dir", "outputs/models"),
        device=config_dict.get("device", "auto"),
        seed=config_dict.get("seed", 42),
    )

    return config


def _serialize_config_dict(config_dict):
    """Convert tuples to lists for YAML serialization."""
    serialized = {}
    for key, value in config_dict.items():
        if isinstance(value, tuple):
            serialized[key] = list(value)
        else:
            serialized[key] = value
    return serialized


def save_config(config: Config, config_path: str):
    """
    Save configuration to YAML file.

    Args:
        config (Config): Configuration object to save
        config_path (str): Path where to save the configuration
    """
    # Convert config to dictionary
    config_dict = {
        "project_name": config.project_name,
        "data": {
            "max_sequence_length": config.data.max_sequence_length,
            "vocab_size": config.data.vocab_size,
            "min_word_frequency": config.data.min_word_frequency,
            "test_size": config.data.test_size,
            "validation_size": config.data.validation_size,
            "random_state": config.data.random_state,
            "stratify": config.data.stratify,
            "text_column": config.data.text_column,
            "label_column": config.data.label_column,
            "lowercase": config.data.lowercase,
            "remove_punctuation": config.data.remove_punctuation,
            "remove_stopwords": config.data.remove_stopwords,
            "stemming": config.data.stemming,
            "lemmatization": config.data.lemmatization,
        },
        "model": _serialize_config_dict(config.model.__dict__),
        "training": config.training.__dict__,
        "cross_validation": config.cross_validation.__dict__,
        "logging": config.logging.__dict__,
        "data_dir": config.data_dir,
        "output_dir": config.output_dir,
        "model_dir": config.model_dir,
        "device": config.device,
        "seed": config.seed,
    }

    # Create directory if it doesn't exist (only if path contains directory)
    config_dir = os.path.dirname(config_path)
    if config_dir:
        os.makedirs(config_dir, exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Configuration saved to {config_path}")


def get_default_config(model_type: str = "lstm") -> Config:
    """
    Get default configuration for a specific model type.

    Args:
        model_type (str): Type of model (lstm, cnn, naive_bayes, svm, logistic_regression)

    Returns:
        Config: Default configuration
    """
    if model_type == "lstm":
        model_config = LSTMConfig()
    elif model_type == "cnn":
        model_config = CNNConfig()
    elif model_type in ["naive_bayes", "svm", "logistic_regression"]:
        model_config = ClassicalMLConfig(model_type=model_type)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    config = Config(model=model_config)
    return config


def setup_directories(config: Config):
    """
    Create necessary directories based on configuration.

    Args:
        config (Config): Configuration object
    """
    directories = [
        config.data_dir,
        config.output_dir,
        config.model_dir,
        config.logging.log_dir,
        config.logging.tensorboard_dir,
        os.path.join(config.data_dir, "raw"),
        os.path.join(config.data_dir, "processed"),
        os.path.join(config.data_dir, "splits"),
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Created directory: {directory}")


if __name__ == "__main__":
    # Test configuration system
    import json

    logging.basicConfig(level=logging.INFO)

    print("Configuration System Test")
    print("=" * 50)

    # Test default configs for different model types
    model_types = ["lstm", "cnn", "naive_bayes", "svm", "logistic_regression"]

    for model_type in model_types:
        print(f"\n{model_type.upper()} Configuration:")
        config = get_default_config(model_type)
        print(f"Model type: {config.model.model_type}")
        print(f"Model config: {config.model}")

    # Test saving and loading
    test_config = get_default_config("lstm")
    test_config_path = "test_config.yaml"

    print(f"\nSaving test configuration to {test_config_path}")
    save_config(test_config, test_config_path)

    print(f"Loading configuration from {test_config_path}")
    loaded_config = load_config(test_config_path)

    print(f"Loaded model type: {loaded_config.model.model_type}")

    # Cleanup
    if os.path.exists(test_config_path):
        os.remove(test_config_path)
        print(f"Cleaned up {test_config_path}")
