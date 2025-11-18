#!/usr/bin/env python3
"""
Main training script for NLP classification models.
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from sklearn.model_selection import train_test_split

from config import load_config, get_default_config
from utils.device import get_device, get_device_info, setup_device_optimization
from data.loader import DataLoader
from data.preprocessor import TextPreprocessor, LabelEncoder
from data.dataset import Vocabulary, TextClassificationDataset, create_data_loaders
from models.classical import (
    NaiveBayesClassifier,
    SVMClassifier,
    LogisticRegressionClassifier,
)
from models.neural import LSTMClassifier, CNNClassifier
from training.trainer import Trainer
from training.evaluator import ModelEvaluator
from training.cross_validator import CrossValidator


def setup_logging(log_level: str = "INFO", log_dir: str = "outputs/logs"):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"training_{int(time.time())}.log")

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_and_preprocess_data(config, logger):
    """Load and preprocess the dataset."""
    logger.info("Loading and preprocessing data...")

    # Check if data file exists
    data_files = []
    data_dir = Path(config.data_dir) / "raw"

    for ext in [".csv", ".tsv", ".json", ".txt"]:
        data_files.extend(list(data_dir.glob(f"*{ext}")))

    if not data_files:
        logger.error(f"No data files found in {data_dir}")
        logger.info("Please place your data files in the data/raw/ directory")
        logger.info("Supported formats: CSV, TSV, JSON, TXT")
        sys.exit(1)

    # Use the first data file found
    data_file = data_files[0]
    logger.info(f"Using data file: {data_file}")

    # Load data
    loader = DataLoader(
        text_column=config.data.text_column, label_column=config.data.label_column
    )

    df = loader.load_data(str(data_file))

    # Validate data
    is_valid, issues = loader.validate_data(df)
    if not is_valid:
        logger.error("Data validation failed:")
        for issue in issues:
            logger.error(f"  - {issue}")
        sys.exit(1)

    # Get data info
    data_info = loader.get_data_info(df)
    logger.info(
        f"Dataset loaded: {data_info['num_samples']} samples, {data_info['label_statistics']['num_unique_labels']} classes"
    )

    # Extract texts and labels
    texts = df[config.data.text_column].tolist()
    labels = df[config.data.label_column].tolist()

    # Preprocess texts
    logger.info("Preprocessing texts...")
    preprocessor = TextPreprocessor(
        lowercase=config.data.lowercase,
        remove_punctuation=config.data.remove_punctuation,
        remove_stopwords=config.data.remove_stopwords,
        stemming=config.data.stemming,
        lemmatization=config.data.lemmatization,
    )

    preprocessed_texts = preprocessor.preprocess_batch(texts)

    # Encode labels
    logger.info("Encoding labels...")
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    logger.info(f"Text preprocessing completed. Classes: {label_encoder.num_classes}")

    return preprocessed_texts, encoded_labels, label_encoder, data_info


def train_classical_model(config, texts, labels, logger):
    """Train a classical ML model."""
    logger.info(f"Training classical model: {config.model.model_type}")

    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts,
        labels,
        test_size=config.data.validation_size,
        random_state=config.seed,
        stratify=labels if config.data.stratify else None,
    )

    # Create model
    if config.model.model_type == "naive_bayes":
        model = NaiveBayesClassifier(
            vectorizer_type=config.model.vectorizer,
            max_features=config.model.max_features,
            ngram_range=tuple(config.model.ngram_range),
            alpha=config.model.alpha,
        )
    elif config.model.model_type == "svm":
        model = SVMClassifier(
            vectorizer_type=config.model.vectorizer,
            max_features=config.model.max_features,
            ngram_range=tuple(config.model.ngram_range),
            C=config.model.svm_C,
            kernel=config.model.svm_kernel,
            gamma=config.model.svm_gamma,
        )
    elif config.model.model_type == "logistic_regression":
        model = LogisticRegressionClassifier(
            vectorizer_type=config.model.vectorizer,
            max_features=config.model.max_features,
            ngram_range=tuple(config.model.ngram_range),
            C=config.model.lr_C,
            penalty=config.model.lr_penalty,
            solver=config.model.lr_solver,
            max_iter=config.model.lr_max_iter,
        )
    else:
        raise ValueError(f"Unknown classical model type: {config.model.model_type}")

    # Train model
    start_time = time.time()
    model.fit(train_texts, [str(label) for label in train_labels])
    train_time = time.time() - start_time

    logger.info(f"Training completed in {train_time:.2f} seconds")

    # Evaluate model
    evaluator = ModelEvaluator()

    # Validation predictions
    val_predictions = model.predict(val_texts)
    val_metrics = evaluator.evaluate_predictions(
        [str(label) for label in val_labels],
        val_predictions.tolist(),
        model_name=config.model.model_type,
    )

    logger.info(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
    logger.info(f"Validation F1 (macro): {val_metrics['macro_f1']:.4f}")

    # Save model
    model_path = os.path.join(
        config.model_dir, f"{config.model.model_type}_model.joblib"
    )
    os.makedirs(config.model_dir, exist_ok=True)
    model.save_model(model_path)

    return model, val_metrics, train_time


def train_neural_model(config, texts, labels, label_encoder, device, logger):
    """Train a neural network model."""
    logger.info(f"Training neural model: {config.model.model_type}")

    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts,
        labels,
        test_size=config.data.validation_size,
        random_state=config.seed,
        stratify=labels if config.data.stratify else None,
    )

    # Build vocabulary
    logger.info("Building vocabulary...")
    vocabulary = Vocabulary(
        vocab_size=config.data.vocab_size, min_frequency=config.data.min_word_frequency
    )
    vocabulary.build_from_texts(train_texts)

    # Create datasets and data loaders
    train_loader, val_loader = create_data_loaders(
        train_texts,
        train_labels,
        val_texts,
        val_labels,
        vocabulary,
        batch_size=config.training.batch_size,
        max_length=config.data.max_sequence_length,
    )

    # Create model
    if config.model.model_type == "lstm":
        model = LSTMClassifier(
            vocab_size=len(vocabulary),
            num_classes=label_encoder.num_classes,
            embedding_dim=config.model.embedding_dim,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
            bidirectional=config.model.bidirectional,
        )
    elif config.model.model_type == "cnn":
        model = CNNClassifier(
            vocab_size=len(vocabulary),
            num_classes=label_encoder.num_classes,
            embedding_dim=config.model.embedding_dim,
            num_filters=config.model.num_filters,
            filter_sizes=config.model.filter_sizes,
            dropout=config.model.dropout,
        )
    else:
        raise ValueError(f"Unknown neural model type: {config.model.model_type}")

    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        optimizer_name=config.training.optimizer,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        scheduler_name=config.training.scheduler,
        gradient_clip_norm=config.training.gradient_clip_norm,
        early_stopping_patience=config.training.early_stopping_patience,
        early_stopping_metric=config.training.early_stopping_metric,
        min_delta=config.training.min_delta,
    )

    # Train model
    logger.info("Starting neural network training...")
    start_time = time.time()

    model_save_path = os.path.join(
        config.model_dir, f"{config.model.model_type}_model.pt"
    )

    trainer.train(
        train_loader,
        val_loader,
        num_epochs=config.training.num_epochs,
        save_best_model=True,
        model_save_path=model_save_path,
    )

    train_time = time.time() - start_time
    logger.info(f"Training completed in {train_time:.2f} seconds")

    # Evaluate model
    val_metrics = trainer.evaluate(val_loader)

    # logger.info(f"Got metrics: {val_metrics}")

    logger.info(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
    logger.info(f"Validation F1 (macro): {val_metrics['macro_f1']:.4f}")

    return trainer, val_metrics, train_time, vocabulary


def run_cross_validation(config, texts, labels, label_encoder, device, logger):
    """Run cross-validation experiment."""
    logger.info("Running cross-validation...")

    cv = CrossValidator(
        n_splits=config.cross_validation.n_splits,
        shuffle=config.cross_validation.shuffle,
        stratify=config.cross_validation.stratify,
        random_state=config.cross_validation.random_state,
        target_accuracy=config.cross_validation.target_accuracy,
    )

    if config.model.model_type in ["naive_bayes", "svm", "logistic_regression"]:
        # Classical model cross-validation
        if config.model.model_type == "naive_bayes":
            model_class = NaiveBayesClassifier
            model_params = {
                "vectorizer_type": config.model.vectorizer,
                "max_features": config.model.max_features,
                "ngram_range": tuple(config.model.ngram_range),
                "alpha": config.model.alpha,
            }
        elif config.model.model_type == "svm":
            model_class = SVMClassifier
            model_params = {
                "vectorizer_type": config.model.vectorizer,
                "max_features": config.model.max_features,
                "ngram_range": tuple(config.model.ngram_range),
                "C": config.model.svm_C,
                "kernel": config.model.svm_kernel,
                "gamma": config.model.svm_gamma,
            }
        elif config.model.model_type == "logistic_regression":
            model_class = LogisticRegressionClassifier
            model_params = {
                "vectorizer_type": config.model.vectorizer,
                "max_features": config.model.max_features,
                "ngram_range": tuple(config.model.ngram_range),
                "C": config.model.lr_C,
                "penalty": config.model.lr_penalty,
                "solver": config.model.lr_solver,
                "max_iter": config.model.lr_max_iter,
            }

        results = cv.cross_validate_classical(
            model_class, model_params, texts, [str(label) for label in labels]
        )

    else:
        # Neural model cross-validation would require more complex setup
        logger.warning(
            "Cross-validation for neural models is not implemented in this script"
        )
        logger.info("Neural models will be trained with simple train-validation split")
        return None

    # Save cross-validation results
    cv_results_path = os.path.join(config.output_dir, "cv_results.json")
    cv.save_results(results, cv_results_path)

    # Check if target accuracy is met
    target_met = results["target_accuracy_met"]
    logger.info(
        f"Cross-validation target accuracy ({config.cross_validation.target_accuracy:.1%}) met: {'✓' if target_met else '✗'}"
    )

    return results


def main():
    parser = argparse.ArgumentParser(description="Train NLP Classification Model")
    parser.add_argument("--config", "-c", type=str, help="Path to configuration file")
    parser.add_argument(
        "--model-type",
        "-m",
        type=str,
        choices=["lstm", "cnn", "naive_bayes", "svm", "logistic_regression"],
        help="Model type to train (overrides config)",
    )
    parser.add_argument(
        "--data-file",
        "-d",
        type=str,
        help="Path to data file (overrides automatic detection)",
    )
    parser.add_argument(
        "--cross-validation",
        "--cv",
        action="store_true",
        help="Run cross-validation instead of single train-test split",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="outputs",
        help="Output directory for models and results",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Load configuration
    if args.config and os.path.exists(args.config):
        config = load_config(args.config)
        print(f"Loaded configuration from {args.config}")
    else:
        model_type = args.model_type or "lstm"
        config = get_default_config(model_type)
        print(f"Using default configuration for {model_type}")

    # Override config with command line arguments
    if args.model_type:
        # Need to recreate config with the new model type
        config = get_default_config(args.model_type)

    if args.output_dir:
        config.output_dir = args.output_dir
        config.model_dir = os.path.join(args.output_dir, "models")
        config.logging.log_dir = os.path.join(args.output_dir, "logs")

    # Setup
    logger = setup_logging(
        log_level="DEBUG" if args.verbose else config.logging.log_level,
        log_dir=config.logging.log_dir,
    )

    set_seed(config.seed)

    # Device setup
    if config.device == "auto":
        device = get_device()
    else:
        device = torch.device(config.device)

    setup_device_optimization(device)

    device_info = get_device_info()
    logger.info(f"Using device: {device}")
    logger.info(f"Available devices: {device_info['available_devices']}")

    try:
        # Load and preprocess data
        texts, labels, label_encoder, data_info = load_and_preprocess_data(
            config, logger
        )

        # Save label encoder
        import joblib

        label_encoder_path = os.path.join(config.output_dir, "label_encoder.joblib")
        os.makedirs(config.output_dir, exist_ok=True)
        joblib.dump(label_encoder, label_encoder_path)
        logger.info(f"Label encoder saved to {label_encoder_path}")

        # Run cross-validation if requested
        if args.cross_validation and config.cross_validation.enabled:
            cv_results = run_cross_validation(
                config, texts, labels, label_encoder, device, logger
            )

            if cv_results and not cv_results["target_accuracy_met"]:
                logger.warning(f"Target accuracy not met in cross-validation!")
                logger.info(
                    "Consider adjusting hyperparameters or trying different models"
                )

        # Train model
        if config.model.model_type in ["naive_bayes", "svm", "logistic_regression"]:
            model, metrics, train_time = train_classical_model(
                config, texts, labels, logger
            )

            # Generate evaluation report
            evaluator = ModelEvaluator(
                class_names=list(label_encoder.id_to_label.values())
            )
            report = evaluator.generate_evaluation_report(metrics)

            # Save report
            report_path = os.path.join(config.output_dir, "evaluation_report.txt")
            with open(report_path, "w") as f:
                f.write(report)

            # Save metrics
            metrics_path = os.path.join(config.output_dir, "metrics.json")
            evaluator.save_metrics(metrics, metrics_path)

        else:
            trainer, metrics, train_time, vocabulary = train_neural_model(
                config, texts, labels, label_encoder, device, logger
            )

            # Save vocabulary
            vocab_path = os.path.join(config.output_dir, "vocabulary.joblib")
            joblib.dump(vocabulary, vocab_path)
            logger.info(f"Vocabulary saved to {vocab_path}")

            # Generate evaluation report
            evaluator = ModelEvaluator(
                class_names=list(label_encoder.id_to_label.values())
            )

            model_object = trainer.model

            metrics["model_name"] = model_object.__class__.__name__

            report = evaluator.generate_evaluation_report(metrics)

            # Save report
            report_path = os.path.join(config.output_dir, "evaluation_report.txt")
            with open(report_path, "w") as f:
                f.write(report)

            # Save metrics
            metrics_path = os.path.join(config.output_dir, "metrics.json")
            evaluator.save_metrics(metrics, metrics_path)

        # Final summary
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Model: {config.model.model_type}")
        logger.info(f"Training time: {train_time:.2f} seconds")
        logger.info(f"Final accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Final F1 (macro): {metrics['macro_f1']:.4f}")

        if args.cross_validation and "cv_results" in locals():
            if cv_results and cv_results["aggregate_results"]:
                cv_acc = cv_results["aggregate_results"]["accuracy_mean"]
                logger.info(f"Cross-validation accuracy: {cv_acc:.4f}")

        logger.info(f"Results saved to: {config.output_dir}")

        # Check target accuracy
        target_met = metrics["accuracy"] >= config.cross_validation.target_accuracy
        logger.info(
            f"Target accuracy ({config.cross_validation.target_accuracy:.1%}) met: {'✓' if target_met else '✗'}"
        )

        if not target_met:
            logger.info("Consider:")
            logger.info("  - Adjusting hyperparameters")
            logger.info("  - Trying different models")
            logger.info("  - Collecting more training data")
            logger.info("  - Feature engineering")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
