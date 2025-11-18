"""
Cross-validation system for NLP classification models.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Union, Optional, Callable
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import torch
from torch.utils.data import DataLoader, Subset
import logging
from tqdm import tqdm
import json
import os

logger = logging.getLogger(__name__)


class CrossValidator:
    """
    Cross-validation system for both classical ML and neural network models.
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        stratify: bool = True,
        random_state: int = 42,
        target_accuracy: float = 0.8,
        scoring_metrics: List[str] = None
    ):
        """
        Initialize CrossValidator.

        Args:
            n_splits (int): Number of cross-validation folds
            shuffle (bool): Whether to shuffle data before splitting
            stratify (bool): Whether to use stratified splits
            random_state (int): Random state for reproducibility
            target_accuracy (float): Target accuracy threshold
            scoring_metrics (List[str]): List of metrics to compute
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.stratify = stratify
        self.random_state = random_state
        self.target_accuracy = target_accuracy

        if scoring_metrics is None:
            scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']
        self.scoring_metrics = scoring_metrics

        # Initialize cross-validation splitter
        if stratify:
            self.cv_splitter = StratifiedKFold(
                n_splits=n_splits,
                shuffle=shuffle,
                random_state=random_state
            )
        else:
            self.cv_splitter = KFold(
                n_splits=n_splits,
                shuffle=shuffle,
                random_state=random_state
            )

        logger.info(f"CrossValidator initialized:")
        logger.info(f"  Folds: {n_splits}")
        logger.info(f"  Stratified: {stratify}")
        logger.info(f"  Target accuracy: {target_accuracy}")

    def cross_validate_classical(
        self,
        model_class: Callable,
        model_params: Dict[str, Any],
        texts: List[str],
        labels: List[str],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Perform cross-validation for classical ML models.

        Args:
            model_class: Classical model class
            model_params (Dict[str, Any]): Model parameters
            texts (List[str]): Input texts
            labels (List[str]): Target labels
            verbose (bool): Whether to show progress

        Returns:
            Dict[str, Any]: Cross-validation results
        """
        logger.info(f"Starting cross-validation for {model_class.__name__}")

        # Convert to numpy arrays
        texts = np.array(texts)
        labels = np.array(labels)

        # Initialize results storage
        fold_results = []
        all_predictions = []
        all_true_labels = []

        # Perform cross-validation
        for fold, (train_idx, val_idx) in enumerate(self.cv_splitter.split(texts, labels)):
            if verbose:
                logger.info(f"Processing fold {fold + 1}/{self.n_splits}")

            # Split data
            train_texts, val_texts = texts[train_idx], texts[val_idx]
            train_labels, val_labels = labels[train_idx], labels[val_idx]

            # Create and train model
            model = model_class(**model_params)
            model.fit(train_texts.tolist(), train_labels.tolist())

            # Make predictions
            predictions = model.predict(val_texts.tolist())

            # Calculate metrics
            fold_metrics = self._calculate_metrics(val_labels, predictions)
            fold_metrics['fold'] = fold + 1
            fold_results.append(fold_metrics)

            # Store predictions for overall analysis
            all_predictions.extend(predictions)
            all_true_labels.extend(val_labels)

            if verbose:
                logger.info(f"  Fold {fold + 1} accuracy: {fold_metrics['accuracy']:.4f}")

        # Calculate aggregate results
        aggregate_results = self._aggregate_results(fold_results)

        # Overall metrics
        overall_metrics = self._calculate_metrics(all_true_labels, all_predictions)

        results = {
            'fold_results': fold_results,
            'aggregate_results': aggregate_results,
            'overall_metrics': overall_metrics,
            'target_accuracy_met': aggregate_results['accuracy_mean'] >= self.target_accuracy,
            'model_type': model_class.__name__,
            'model_params': model_params,
            'cv_config': {
                'n_splits': self.n_splits,
                'stratify': self.stratify,
                'random_state': self.random_state
            }
        }

        if verbose:
            self._print_results_summary(results)

        return results

    def cross_validate_neural(
        self,
        model_class: Callable,
        model_params: Dict[str, Any],
        dataset,
        trainer_class: Callable,
        trainer_params: Dict[str, Any],
        device: torch.device,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Perform cross-validation for neural network models.

        Args:
            model_class: Neural model class
            model_params (Dict[str, Any]): Model parameters
            dataset: PyTorch dataset
            trainer_class: Trainer class
            trainer_params (Dict[str, Any]): Trainer parameters
            device (torch.device): Device to run on
            verbose (bool): Whether to show progress

        Returns:
            Dict[str, Any]: Cross-validation results
        """
        logger.info(f"Starting cross-validation for {model_class.__name__}")

        # Get labels for stratification
        labels = [dataset[i]['labels'].item() for i in range(len(dataset))]
        indices = np.arange(len(dataset))

        # Initialize results storage
        fold_results = []
        all_predictions = []
        all_true_labels = []

        # Perform cross-validation
        for fold, (train_idx, val_idx) in enumerate(self.cv_splitter.split(indices, labels)):
            if verbose:
                logger.info(f"Processing fold {fold + 1}/{self.n_splits}")

            # Create data loaders for this fold
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            train_loader = DataLoader(
                train_subset,
                batch_size=trainer_params.get('batch_size', 32),
                shuffle=True,
                num_workers=0
            )

            val_loader = DataLoader(
                val_subset,
                batch_size=trainer_params.get('batch_size', 32),
                shuffle=False,
                num_workers=0
            )

            # Create and train model
            model = model_class(**model_params).to(device)
            trainer = trainer_class(model, device, **trainer_params)

            # Train model
            trainer.train(train_loader, val_loader, verbose=False)

            # Evaluate model
            val_metrics = trainer.evaluate(val_loader)

            # Get predictions for detailed analysis
            predictions, true_labels = trainer.predict(val_loader)

            # Store results
            fold_metrics = {
                'fold': fold + 1,
                'accuracy': val_metrics['accuracy'],
                'precision': val_metrics.get('precision', 0.0),
                'recall': val_metrics.get('recall', 0.0),
                'f1': val_metrics.get('f1', 0.0),
                'loss': val_metrics.get('loss', 0.0)
            }
            fold_results.append(fold_metrics)

            # Store predictions for overall analysis
            all_predictions.extend(predictions)
            all_true_labels.extend(true_labels)

            if verbose:
                logger.info(f"  Fold {fold + 1} accuracy: {fold_metrics['accuracy']:.4f}")

        # Calculate aggregate results
        aggregate_results = self._aggregate_results(fold_results)

        # Overall metrics
        overall_metrics = self._calculate_metrics(all_true_labels, all_predictions)

        results = {
            'fold_results': fold_results,
            'aggregate_results': aggregate_results,
            'overall_metrics': overall_metrics,
            'target_accuracy_met': aggregate_results['accuracy_mean'] >= self.target_accuracy,
            'model_type': model_class.__name__,
            'model_params': model_params,
            'trainer_params': trainer_params,
            'cv_config': {
                'n_splits': self.n_splits,
                'stratify': self.stratify,
                'random_state': self.random_state
            }
        }

        if verbose:
            self._print_results_summary(results)

        return results

    def _calculate_metrics(self, y_true: List, y_pred: List) -> Dict[str, float]:
        """
        Calculate evaluation metrics.

        Args:
            y_true (List): True labels
            y_pred (List): Predicted labels

        Returns:
            Dict[str, float]: Calculated metrics
        """
        metrics = {}

        # Accuracy
        metrics['accuracy'] = accuracy_score(y_true, y_pred)

        # Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )

        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1

        # Macro averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )

        metrics['macro_precision'] = macro_precision
        metrics['macro_recall'] = macro_recall
        metrics['macro_f1'] = macro_f1

        return metrics

    def _aggregate_results(self, fold_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Aggregate results across folds.

        Args:
            fold_results (List[Dict[str, Any]]): Results from each fold

        Returns:
            Dict[str, float]: Aggregated results
        """
        aggregate = {}

        # Get all metric names (excluding 'fold')
        metric_names = [key for key in fold_results[0].keys() if key != 'fold']

        for metric in metric_names:
            values = [fold[metric] for fold in fold_results]
            aggregate[f'{metric}_mean'] = np.mean(values)
            aggregate[f'{metric}_std'] = np.std(values)
            aggregate[f'{metric}_min'] = np.min(values)
            aggregate[f'{metric}_max'] = np.max(values)

        return aggregate

    def _print_results_summary(self, results: Dict[str, Any]):
        """
        Print a summary of cross-validation results.

        Args:
            results (Dict[str, Any]): Cross-validation results
        """
        logger.info("\nCross-Validation Results Summary:")
        logger.info("=" * 50)

        aggregate = results['aggregate_results']

        # Main metrics
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            mean_key = f'{metric}_mean'
            std_key = f'{metric}_std'
            if mean_key in aggregate:
                mean_val = aggregate[mean_key]
                std_val = aggregate[std_key]
                logger.info(f"{metric.capitalize()}: {mean_val:.4f} (±{std_val:.4f})")

        # Target accuracy check
        target_met = results['target_accuracy_met']
        logger.info(f"\nTarget accuracy ({self.target_accuracy:.1%}) met: {'✓' if target_met else '✗'}")

        # Fold-by-fold results
        logger.info(f"\nFold-by-fold results:")
        for fold_result in results['fold_results']:
            fold_num = fold_result['fold']
            accuracy = fold_result['accuracy']
            logger.info(f"  Fold {fold_num}: {accuracy:.4f}")

    def save_results(self, results: Dict[str, Any], filepath: str):
        """
        Save cross-validation results to file.

        Args:
            results (Dict[str, Any]): Results to save
            filepath (str): File path to save results
        """
        # Convert numpy types to Python types for JSON serialization
        results_serializable = self._make_serializable(results)

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2)

        logger.info(f"Cross-validation results saved to {filepath}")

    def load_results(self, filepath: str) -> Dict[str, Any]:
        """
        Load cross-validation results from file.

        Args:
            filepath (str): File path to load results from

        Returns:
            Dict[str, Any]: Loaded results
        """
        with open(filepath, 'r') as f:
            results = json.load(f)

        logger.info(f"Cross-validation results loaded from {filepath}")
        return results

    def _make_serializable(self, obj):
        """
        Convert numpy types to Python types for JSON serialization.

        Args:
            obj: Object to convert

        Returns:
            Serializable object
        """
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        else:
            return obj

    def compare_models(self, results_list: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare results from multiple models.

        Args:
            results_list (List[Dict[str, Any]]): List of cross-validation results

        Returns:
            pd.DataFrame: Comparison table
        """
        comparison_data = []

        for results in results_list:
            model_type = results['model_type']
            aggregate = results['aggregate_results']
            target_met = results['target_accuracy_met']

            row = {
                'Model': model_type,
                'Accuracy_Mean': aggregate.get('accuracy_mean', 0),
                'Accuracy_Std': aggregate.get('accuracy_std', 0),
                'Precision_Mean': aggregate.get('precision_mean', 0),
                'Recall_Mean': aggregate.get('recall_mean', 0),
                'F1_Mean': aggregate.get('f1_mean', 0),
                'Target_Met': target_met
            }

            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Accuracy_Mean', ascending=False)

        return comparison_df


def run_cross_validation_experiment(
    models_config: List[Dict[str, Any]],
    texts: List[str],
    labels: List[str],
    cv_config: Dict[str, Any] = None,
    save_dir: str = "outputs/cv_results"
) -> Dict[str, Any]:
    """
    Run cross-validation experiment for multiple models.

    Args:
        models_config (List[Dict[str, Any]]): Configuration for models to test
        texts (List[str]): Input texts
        labels (List[str]): Target labels
        cv_config (Dict[str, Any]): Cross-validation configuration
        save_dir (str): Directory to save results

    Returns:
        Dict[str, Any]: Experiment results
    """
    if cv_config is None:
        cv_config = {'n_splits': 5, 'random_state': 42}

    # Initialize cross-validator
    cv = CrossValidator(**cv_config)

    # Run cross-validation for each model
    all_results = {}
    for model_config in models_config:
        model_name = model_config['name']
        model_class = model_config['class']
        model_params = model_config['params']

        logger.info(f"\nTesting {model_name}...")

        if hasattr(model_class, 'fit'):  # Classical ML model
            results = cv.cross_validate_classical(
                model_class, model_params, texts, labels
            )
        else:  # Neural network model
            # This would require additional setup for neural models
            logger.warning(f"Neural model cross-validation not implemented in this example")
            continue

        all_results[model_name] = results

        # Save individual results
        os.makedirs(save_dir, exist_ok=True)
        results_file = os.path.join(save_dir, f"{model_name}_cv_results.json")
        cv.save_results(results, results_file)

    # Create comparison
    results_list = list(all_results.values())
    comparison_df = cv.compare_models(results_list)

    # Save comparison
    comparison_file = os.path.join(save_dir, "model_comparison.csv")
    comparison_df.to_csv(comparison_file, index=False)

    experiment_results = {
        'individual_results': all_results,
        'comparison': comparison_df,
        'cv_config': cv_config,
        'summary': {
            'best_model': comparison_df.iloc[0]['Model'],
            'best_accuracy': comparison_df.iloc[0]['Accuracy_Mean'],
            'models_meeting_target': sum(comparison_df['Target_Met'])
        }
    }

    logger.info(f"\nExperiment Summary:")
    logger.info(f"Best model: {experiment_results['summary']['best_model']}")
    logger.info(f"Best accuracy: {experiment_results['summary']['best_accuracy']:.4f}")
    logger.info(f"Models meeting target: {experiment_results['summary']['models_meeting_target']}/{len(models_config)}")

    return experiment_results


if __name__ == "__main__":
    # Test cross-validation system
    logging.basicConfig(level=logging.INFO)

    print("Cross-Validation System Test")
    print("=" * 50)

    # Create sample data
    from src.data.loader import create_sample_data
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_file = f.name

    try:
        create_sample_data(sample_file, num_samples=200, num_classes=3)

        # Load data
        from src.data.loader import DataLoader
        loader = DataLoader()
        df = loader.load_data(sample_file)

        texts = df['text'].tolist()
        labels = df['label'].tolist()

        # Test classical model cross-validation
        from src.models.classical import NaiveBayesClassifier

        cv = CrossValidator(n_splits=3, target_accuracy=0.7)  # Smaller for testing

        model_params = {'vectorizer_type': 'tfidf', 'alpha': 1.0}
        results = cv.cross_validate_classical(
            NaiveBayesClassifier, model_params, texts, labels
        )

        print(f"Cross-validation completed successfully!")
        print(f"Target accuracy met: {results['target_accuracy_met']}")

    finally:
        # Cleanup
        if os.path.exists(sample_file):
            os.remove(sample_file)