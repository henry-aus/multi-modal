"""
Metrics calculation utilities for model evaluation
"""

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate accuracy for a batch of predictions

    Args:
        outputs: Model predictions (logits)
        targets: Ground truth labels

    Returns:
        Accuracy as a float
    """
    _, predicted = torch.max(outputs, 1)
    total = targets.size(0)
    correct = (predicted == targets).sum().item()
    return correct / total


def top_k_accuracy(outputs: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    """
    Calculate top-k accuracy

    Args:
        outputs: Model predictions (logits)
        targets: Ground truth labels
        k: Top-k value

    Returns:
        Top-k accuracy as a float
    """
    _, topk_preds = outputs.topk(k, dim=1, largest=True, sorted=True)
    correct = topk_preds.eq(targets.view(-1, 1).expand_as(topk_preds))
    return correct.float().sum().item() / targets.size(0)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name: str, fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class MetricsCalculator:
    """Calculate and store various metrics for model evaluation"""

    def __init__(self, num_classes: int, class_names: List[str] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class {i}" for i in range(num_classes)]
        self.reset()

    def reset(self):
        """Reset all stored predictions and targets"""
        self.all_predictions = []
        self.all_targets = []

    def update(self, outputs: torch.Tensor, targets: torch.Tensor):
        """
        Update with new batch of predictions and targets

        Args:
            outputs: Model predictions (logits)
            targets: Ground truth labels
        """
        _, predicted = torch.max(outputs, 1)
        self.all_predictions.extend(predicted.cpu().numpy())
        self.all_targets.extend(targets.cpu().numpy())

    def compute_metrics(self) -> Dict:
        """
        Compute all metrics

        Returns:
            Dictionary containing various metrics
        """
        if not self.all_predictions:
            return {}

        predictions = np.array(self.all_predictions)
        targets = np.array(self.all_targets)

        # Basic accuracy
        accuracy = accuracy_score(targets, predictions)

        # Classification report
        report = classification_report(
            targets, predictions,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )

        # Confusion matrix
        cm = confusion_matrix(targets, predictions)

        # Per-class accuracy
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        per_class_accuracy = np.nan_to_num(per_class_accuracy)

        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'per_class_accuracy': per_class_accuracy,
            'macro_avg_precision': report['macro avg']['precision'],
            'macro_avg_recall': report['macro avg']['recall'],
            'macro_avg_f1': report['macro avg']['f1-score'],
            'weighted_avg_precision': report['weighted avg']['precision'],
            'weighted_avg_recall': report['weighted avg']['recall'],
            'weighted_avg_f1': report['weighted avg']['f1-score']
        }

    def plot_confusion_matrix(self, save_path: str = None, normalize: bool = False):
        """
        Plot confusion matrix

        Args:
            save_path: Path to save the plot
            normalize: Whether to normalize the confusion matrix
        """
        metrics = self.compute_metrics()
        if 'confusion_matrix' not in metrics:
            return

        cm = metrics['confusion_matrix']

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt=fmt, cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def print_metrics(self):
        """Print formatted metrics summary"""
        metrics = self.compute_metrics()
        if not metrics:
            print("No metrics to display")
            return

        print(f"\n{'='*50}")
        print(f"EVALUATION METRICS")
        print(f"{'='*50}")
        print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro Avg Precision: {metrics['macro_avg_precision']:.4f}")
        print(f"Macro Avg Recall: {metrics['macro_avg_recall']:.4f}")
        print(f"Macro Avg F1-Score: {metrics['macro_avg_f1']:.4f}")
        print(f"Weighted Avg Precision: {metrics['weighted_avg_precision']:.4f}")
        print(f"Weighted Avg Recall: {metrics['weighted_avg_recall']:.4f}")
        print(f"Weighted Avg F1-Score: {metrics['weighted_avg_f1']:.4f}")

        print(f"\n{'='*50}")
        print(f"PER-CLASS ACCURACY")
        print(f"{'='*50}")
        for i, (class_name, acc) in enumerate(zip(self.class_names, metrics['per_class_accuracy'])):
            print(f"{class_name}: {acc:.4f}")

        print(f"\n{'='*50}")
        print(f"CLASSIFICATION REPORT")
        print(f"{'='*50}")
        report = metrics['classification_report']
        for class_name in self.class_names:
            if class_name in report:
                cls_metrics = report[class_name]
                print(f"{class_name:12} - Precision: {cls_metrics['precision']:.4f}, "
                      f"Recall: {cls_metrics['recall']:.4f}, "
                      f"F1-Score: {cls_metrics['f1-score']:.4f}, "
                      f"Support: {cls_metrics['support']}")


def save_metrics_to_file(metrics: Dict, filepath: str):
    """
    Save metrics to a text file

    Args:
        metrics: Dictionary of computed metrics
        filepath: Path to save the metrics file
    """
    import json

    # Convert numpy arrays to lists for JSON serialization
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            serializable_metrics[key] = value.tolist()
        elif key == 'classification_report':
            # Keep classification report as is
            serializable_metrics[key] = value
        else:
            serializable_metrics[key] = value

    with open(filepath, 'w') as f:
        json.dump(serializable_metrics, f, indent=4)