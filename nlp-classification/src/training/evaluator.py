"""
Evaluation framework for NLP classification models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional, Union
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
import torch
import logging
import os
import json

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive evaluation framework for classification models.
    """

    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Initialize ModelEvaluator.

        Args:
            class_names (List[str], optional): Names of classes for better visualization
        """
        self.class_names = class_names

    def evaluate_predictions(
        self,
        y_true: List[int],
        y_pred: List[int],
        y_proba: Optional[List[List[float]]] = None,
        model_name: str = "Model",
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of model predictions.

        Args:
            y_true (List[int]): True labels
            y_pred (List[int]): Predicted labels
            y_proba (List[List[float]], optional): Prediction probabilities
            model_name (str): Name of the model for reporting

        Returns:
            Dict[str, Any]: Comprehensive evaluation metrics
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        if y_proba is not None:
            y_proba = np.array(y_proba)

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        # Averaged metrics
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )

        weighted_precision, weighted_recall, weighted_f1, _ = (
            precision_recall_fscore_support(
                y_true, y_pred, average="weighted", zero_division=0
            )
        )

        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="micro", zero_division=0
        )

        # Confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)

        # Classification report
        class_names = self.class_names or [
            str(i) for i in range(len(np.unique(y_true)))
        ]
        classification_rep = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
        )

        # Compile basic metrics
        metrics = {
            "model_name": model_name,
            "accuracy": accuracy,
            "precision_per_class": precision.tolist(),
            "recall_per_class": recall.tolist(),
            "f1_per_class": f1.tolist(),
            "support_per_class": support.tolist(),
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall,
            "weighted_f1": weighted_f1,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "confusion_matrix": conf_matrix.tolist(),
            "classification_report": classification_rep,
            "num_samples": len(y_true),
            "num_classes": len(np.unique(y_true)),
        }

        # Add probability-based metrics if available
        if y_proba is not None:
            metrics.update(self._calculate_probability_metrics(y_true, y_proba))

        logger.info(f"Evaluation completed for {model_name}")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Macro F1: {macro_f1:.4f}")
        logger.info(f"  Weighted F1: {weighted_f1:.4f}")

        return metrics

    def _calculate_probability_metrics(
        self, y_true: np.ndarray, y_proba: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculate probability-based metrics.

        Args:
            y_true (np.ndarray): True labels
            y_proba (np.ndarray): Prediction probabilities

        Returns:
            Dict[str, Any]: Probability-based metrics
        """
        num_classes = y_proba.shape[1]
        prob_metrics = {}

        if num_classes == 2:
            # Binary classification
            auc_score = roc_auc_score(y_true, y_proba[:, 1])
            avg_precision = average_precision_score(y_true, y_proba[:, 1])

            prob_metrics.update(
                {"roc_auc": auc_score, "average_precision": avg_precision}
            )

        elif num_classes > 2:
            # Multi-class classification
            try:
                # One-vs-Rest AUC
                auc_ovr = roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="macro"
                )
                # One-vs-One AUC
                auc_ovo = roc_auc_score(
                    y_true, y_proba, multi_class="ovo", average="macro"
                )

                prob_metrics.update({"roc_auc_ovr": auc_ovr, "roc_auc_ovo": auc_ovo})
            except ValueError as e:
                logger.warning(f"Could not calculate AUC scores: {e}")

        # Confidence-based metrics
        max_proba = np.max(y_proba, axis=1)
        prob_metrics.update(
            {
                "mean_confidence": np.mean(max_proba),
                "median_confidence": np.median(max_proba),
                "confidence_std": np.std(max_proba),
                "min_confidence": np.min(max_proba),
                "max_confidence": np.max(max_proba),
            }
        )

        return prob_metrics

    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        save_path: Optional[str] = None,
        title: str = "Confusion Matrix",
        figsize: Tuple[int, int] = (8, 6),
    ) -> plt.Figure:
        """
        Plot confusion matrix.

        Args:
            confusion_matrix (np.ndarray): Confusion matrix
            save_path (str, optional): Path to save the plot
            title (str): Plot title
            figsize (Tuple[int, int]): Figure size

        Returns:
            plt.Figure: Matplotlib figure
        """
        plt.figure(figsize=figsize)

        # Use class names if available
        if self.class_names:
            labels = self.class_names[: confusion_matrix.shape[0]]
        else:
            labels = [f"Class {i}" for i in range(confusion_matrix.shape[0])]

        # Create heatmap
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={"label": "Count"},
        )

        plt.title(title)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Confusion matrix saved to {save_path}")

        return plt.gcf()

    def plot_roc_curves(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        save_path: Optional[str] = None,
        title: str = "ROC Curves",
        figsize: Tuple[int, int] = (10, 8),
    ) -> plt.Figure:
        """
        Plot ROC curves for multi-class classification.

        Args:
            y_true (np.ndarray): True labels
            y_proba (np.ndarray): Prediction probabilities
            save_path (str, optional): Path to save the plot
            title (str): Plot title
            figsize (Tuple[int, int]): Figure size

        Returns:
            plt.Figure: Matplotlib figure
        """
        from sklearn.preprocessing import label_binarize

        num_classes = y_proba.shape[1]

        plt.figure(figsize=figsize)

        if num_classes == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            auc_score = roc_auc_score(y_true, y_proba[:, 1])

            plt.plot(fpr, tpr, linewidth=2, label=f"ROC curve (AUC = {auc_score:.3f})")

        else:
            # Multi-class classification
            # Binarize the output
            y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))

            # Compute ROC curve and AUC for each class
            for i in range(num_classes):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                auc_score = roc_auc_score(y_true_bin[:, i], y_proba[:, i])

                class_name = self.class_names[i] if self.class_names else f"Class {i}"
                plt.plot(
                    fpr, tpr, linewidth=2, label=f"{class_name} (AUC = {auc_score:.3f})"
                )

        # Plot diagonal line
        plt.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"ROC curves saved to {save_path}")

        return plt.gcf()

    def plot_precision_recall_curves(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        save_path: Optional[str] = None,
        title: str = "Precision-Recall Curves",
        figsize: Tuple[int, int] = (10, 8),
    ) -> plt.Figure:
        """
        Plot Precision-Recall curves for multi-class classification.

        Args:
            y_true (np.ndarray): True labels
            y_proba (np.ndarray): Prediction probabilities
            save_path (str, optional): Path to save the plot
            title (str): Plot title
            figsize (Tuple[int, int]): Figure size

        Returns:
            plt.Figure: Matplotlib figure
        """
        from sklearn.preprocessing import label_binarize

        num_classes = y_proba.shape[1]

        plt.figure(figsize=figsize)

        if num_classes == 2:
            # Binary classification
            precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
            avg_precision = average_precision_score(y_true, y_proba[:, 1])

            plt.plot(
                recall,
                precision,
                linewidth=2,
                label=f"PR curve (AP = {avg_precision:.3f})",
            )

        else:
            # Multi-class classification
            # Binarize the output
            y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))

            # Compute PR curve and AP for each class
            for i in range(num_classes):
                precision, recall, _ = precision_recall_curve(
                    y_true_bin[:, i], y_proba[:, i]
                )
                avg_precision = average_precision_score(y_true_bin[:, i], y_proba[:, i])

                class_name = self.class_names[i] if self.class_names else f"Class {i}"
                plt.plot(
                    recall,
                    precision,
                    linewidth=2,
                    label=f"{class_name} (AP = {avg_precision:.3f})",
                )

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(title)
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Precision-Recall curves saved to {save_path}")

        return plt.gcf()

    def plot_class_distribution(
        self,
        y_true: List[int],
        save_path: Optional[str] = None,
        title: str = "Class Distribution",
        figsize: Tuple[int, int] = (10, 6),
    ) -> plt.Figure:
        """
        Plot class distribution.

        Args:
            y_true (List[int]): True labels
            save_path (str, optional): Path to save the plot
            title (str): Plot title
            figsize (Tuple[int, int]): Figure size

        Returns:
            plt.Figure: Matplotlib figure
        """
        plt.figure(figsize=figsize)

        # Count classes
        unique, counts = np.unique(y_true, return_counts=True)

        # Use class names if available
        if self.class_names:
            labels = [self.class_names[i] for i in unique]
        else:
            labels = [f"Class {i}" for i in unique]

        # Create bar plot
        bars = plt.bar(labels, counts, alpha=0.7, color="skyblue", edgecolor="black")

        # Add count labels on bars
        for bar, count in zip(bars, counts):
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.1,
                str(count),
                ha="center",
                va="bottom",
            )

        plt.xlabel("Classes")
        plt.ylabel("Number of Samples")
        plt.title(title)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Class distribution plot saved to {save_path}")

        return plt.gcf()

    def generate_evaluation_report(
        self, metrics: Dict[str, Any], save_path: Optional[str] = None
    ) -> str:
        """
        Generate a comprehensive evaluation report.

        Args:
            metrics (Dict[str, Any]): Evaluation metrics
            save_path (str, optional): Path to save the report

        Returns:
            str: Formatted evaluation report
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append(f"EVALUATION REPORT: {metrics['model_name']}")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Overall performance
        report_lines.append("OVERALL PERFORMANCE:")
        report_lines.append("-" * 40)
        report_lines.append(f"Accuracy:           {metrics['accuracy']:.4f}")
        report_lines.append(f"Macro Precision:    {metrics['macro_precision']:.4f}")
        report_lines.append(f"Macro Recall:       {metrics['macro_recall']:.4f}")
        report_lines.append(f"Macro F1-Score:     {metrics['macro_f1']:.4f}")
        report_lines.append(f"Weighted Precision: {metrics['weighted_precision']:.4f}")
        report_lines.append(f"Weighted Recall:    {metrics['weighted_recall']:.4f}")
        report_lines.append(f"Weighted F1-Score:  {metrics['weighted_f1']:.4f}")
        report_lines.append("")

        # Dataset info
        # report_lines.append("DATASET INFORMATION:")
        # report_lines.append("-" * 40)
        # report_lines.append(f"Number of samples:  {metrics['num_samples']}")
        # report_lines.append(f"Number of classes:  {metrics['num_classes']}")
        # report_lines.append("")

        # Per-class performance
        report_lines.append("PER-CLASS PERFORMANCE:")
        report_lines.append("-" * 40)
        report_lines.append(
            f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}"
        )
        report_lines.append("-" * 60)

        for i in range(len(metrics["precision_per_class"])):
            class_name = self.class_names[i] if self.class_names else f"Class {i}"
            precision = metrics["precision_per_class"][i]
            recall = metrics["recall_per_class"][i]
            f1 = metrics["f1_per_class"][i]
            support = metrics["support_per_class"][i]

            report_lines.append(
                f"{class_name:<15} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {support:<10}"
            )
        report_lines.append("")

        # Probability-based metrics (if available)
        if "roc_auc" in metrics or "roc_auc_ovr" in metrics:
            report_lines.append("PROBABILITY-BASED METRICS:")
            report_lines.append("-" * 40)

            if "roc_auc" in metrics:
                report_lines.append(f"ROC AUC:            {metrics['roc_auc']:.4f}")
                report_lines.append(
                    f"Average Precision:  {metrics['average_precision']:.4f}"
                )

            if "roc_auc_ovr" in metrics:
                report_lines.append(f"ROC AUC (OvR):      {metrics['roc_auc_ovr']:.4f}")
                report_lines.append(f"ROC AUC (OvO):      {metrics['roc_auc_ovo']:.4f}")

            report_lines.append(f"Mean Confidence:    {metrics['mean_confidence']:.4f}")
            report_lines.append(
                f"Median Confidence:  {metrics['median_confidence']:.4f}"
            )
            report_lines.append("")

        # Confusion matrix summary
        conf_matrix = np.array(metrics["confusion_matrix"])
        report_lines.append("CONFUSION MATRIX:")
        report_lines.append("-" * 40)

        # Header
        if self.class_names:
            header = "Actual\\Predicted" + "".join(
                [f"{name:>8}" for name in self.class_names[: conf_matrix.shape[1]]]
            )
        else:
            header = "Actual\\Predicted" + "".join(
                [f"C{i:>7}" for i in range(conf_matrix.shape[1])]
            )
        report_lines.append(header)

        # Matrix rows
        for i, row in enumerate(conf_matrix):
            class_name = self.class_names[i] if self.class_names else f"Class {i}"
            row_str = f"{class_name:<16}" + "".join([f"{val:>8}" for val in row])
            report_lines.append(row_str)

        report_lines.append("")
        report_lines.append("=" * 80)

        report_text = "\n".join(report_lines)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w") as f:
                f.write(report_text)
            logger.info(f"Evaluation report saved to {save_path}")

        return report_text

    def save_metrics(self, metrics: Dict[str, Any], filepath: str):
        """
        Save metrics to JSON file.

        Args:
            metrics (Dict[str, Any]): Metrics to save
            filepath (str): File path to save metrics
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_metrics = self._make_json_serializable(metrics)

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(serializable_metrics, f, indent=2)

        logger.info(f"Metrics saved to {filepath}")

    def _make_json_serializable(self, obj):
        """Make object JSON serializable."""
        if isinstance(obj, dict):
            return {
                key: self._make_json_serializable(value) for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
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
        Compare multiple models' performance.

        Args:
            results_list (List[Dict[str, Any]]): List of evaluation results

        Returns:
            pd.DataFrame: Comparison table
        """
        comparison_data = []

        for result in results_list:
            row = {
                "Model": result["model_name"],
                "Accuracy": result["accuracy"],
                "Macro_Precision": result["macro_precision"],
                "Macro_Recall": result["macro_recall"],
                "Macro_F1": result["macro_f1"],
                "Weighted_F1": result["weighted_f1"],
                "Num_Samples": result["num_samples"],
            }

            # Add AUC if available
            if "roc_auc" in result:
                row["ROC_AUC"] = result["roc_auc"]
            elif "roc_auc_ovr" in result:
                row["ROC_AUC_OvR"] = result["roc_auc_ovr"]

            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values("Accuracy", ascending=False)

        return comparison_df


if __name__ == "__main__":
    # Test evaluation framework
    import logging

    logging.basicConfig(level=logging.INFO)

    print("Evaluation Framework Test")
    print("=" * 50)

    # Create sample data
    np.random.seed(42)
    y_true = np.random.choice([0, 1, 2], size=100)
    y_pred = y_true.copy()
    # Add some noise
    noise_indices = np.random.choice(len(y_pred), size=10, replace=False)
    y_pred[noise_indices] = np.random.choice([0, 1, 2], size=10)

    # Sample probabilities
    y_proba = np.random.dirichlet([1, 1, 1], size=100)

    # Test evaluator
    evaluator = ModelEvaluator(class_names=["Class A", "Class B", "Class C"])

    # Evaluate
    metrics = evaluator.evaluate_predictions(
        y_true.tolist(), y_pred.tolist(), y_proba.tolist(), "Test Model"
    )

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")

    # Generate report
    report = evaluator.generate_evaluation_report(metrics)
    print("\nSample Report (first 500 chars):")
    print(report[:500] + "...")

    print("\nEvaluation framework working!")
