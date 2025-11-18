"""
Evaluation script for CNN image classification
Evaluates trained model and generates comprehensive metrics and visualizations
"""

import os
import sys
import yaml
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from tqdm import tqdm

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.dataset import create_data_loaders
from src.models.resnet import create_model
from src.utils.device import get_device, get_device_info
from src.utils.metrics import MetricsCalculator, save_metrics_to_file


class Evaluator:
    """Model evaluator for CNN image classification"""

    def __init__(self, config_path: str, checkpoint_path: str):
        """
        Initialize evaluator

        Args:
            config_path: Path to configuration YAML file
            checkpoint_path: Path to model checkpoint
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.checkpoint_path = checkpoint_path

        # Setup device
        self.device = get_device()
        print(f"Using device: {self.device}")

        # Create results directory
        results_dir = Path(self.config['logging']['results_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = results_dir

    def load_model(self):
        """Load trained model from checkpoint"""
        print(f"Loading model from: {self.checkpoint_path}")

        # Create model
        model = create_model(
            num_classes=self.config['data']['num_classes'],
            pretrained=False,  # Don't need pretrained weights when loading checkpoint
            dropout_rate=self.config['model']['dropout_rate'],
            device=self.device
        )

        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])

        # Print checkpoint information
        print(f"Loaded checkpoint from epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"Checkpoint accuracy: {checkpoint.get('accuracy', 'unknown'):.2f}%")

        return model

    def create_data_loader(self):
        """Create test data loader"""
        print("Creating test data loader...")
        data_config = self.config['data']

        _, test_loader = create_data_loaders(
            train_file=data_config['train_file'],
            test_file=data_config['test_file'],
            data_root=data_config['data_root'],
            batch_size=data_config['batch_size'],
            num_workers=data_config['num_workers'],
            image_size=data_config['image_size']
        )

        print(f"Test samples: {len(test_loader.dataset)}")
        print(f"Test batches: {len(test_loader)}")

        return test_loader

    def evaluate(self, model, test_loader):
        """
        Evaluate model on test set

        Args:
            model: Trained model
            test_loader: Test data loader

        Returns:
            Dictionary containing evaluation metrics
        """
        model.eval()

        # Initialize metrics calculator
        class_names = [f"Class {i+1}" for i in range(self.config['data']['num_classes'])]
        metrics_calc = MetricsCalculator(self.config['data']['num_classes'], class_names)

        # Store predictions for detailed analysis
        all_predictions = []
        all_probabilities = []
        all_targets = []
        all_image_paths = []

        print("Evaluating model...")
        with torch.no_grad():
            pbar = tqdm(test_loader, desc='Evaluation')

            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                outputs = model(inputs)
                probabilities = torch.softmax(outputs, dim=1)

                # Update metrics calculator
                metrics_calc.update(outputs, targets)

                # Store detailed predictions
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

                # Update progress bar
                current_acc = (predicted == targets).float().mean().item() * 100
                pbar.set_postfix({'Batch Acc': f'{current_acc:.2f}%'})

        # Compute comprehensive metrics
        metrics = metrics_calc.compute_metrics()

        # Add additional information
        evaluation_results = {
            'timestamp': datetime.now().isoformat(),
            'checkpoint_path': self.checkpoint_path,
            'config': self.config,
            'metrics': metrics,
            'detailed_predictions': {
                'predictions': all_predictions,
                'probabilities': all_probabilities,
                'targets': all_targets
            }
        }

        return evaluation_results, metrics_calc

    def save_results(self, results, metrics_calc):
        """
        Save evaluation results and generate visualizations

        Args:
            results: Evaluation results dictionary
            metrics_calc: MetricsCalculator instance
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results to JSON
        results_file = self.results_dir / f"evaluation_results_{timestamp}.json"

        # Prepare results for JSON serialization (remove non-serializable items)
        json_results = {
            'timestamp': results['timestamp'],
            'checkpoint_path': results['checkpoint_path'],
            'overall_accuracy': results['metrics']['accuracy'],
            'macro_avg_precision': results['metrics']['macro_avg_precision'],
            'macro_avg_recall': results['metrics']['macro_avg_recall'],
            'macro_avg_f1': results['metrics']['macro_avg_f1'],
            'weighted_avg_precision': results['metrics']['weighted_avg_precision'],
            'weighted_avg_recall': results['metrics']['weighted_avg_recall'],
            'weighted_avg_f1': results['metrics']['weighted_avg_f1'],
            'per_class_accuracy': results['metrics']['per_class_accuracy'],
            'classification_report': results['metrics']['classification_report']
        }

        save_metrics_to_file(json_results, str(results_file))
        print(f"Detailed results saved to: {results_file}")

        # Save confusion matrix plot
        if self.config['evaluation'].get('save_confusion_matrix', True):
            cm_file = self.results_dir / f"confusion_matrix_{timestamp}.png"
            metrics_calc.plot_confusion_matrix(str(cm_file), normalize=False)
            print(f"Confusion matrix saved to: {cm_file}")

            # Also save normalized confusion matrix
            cm_norm_file = self.results_dir / f"confusion_matrix_normalized_{timestamp}.png"
            metrics_calc.plot_confusion_matrix(str(cm_norm_file), normalize=True)
            print(f"Normalized confusion matrix saved to: {cm_norm_file}")

        # Save predictions
        if self.config['evaluation'].get('save_predictions', True):
            pred_file = self.results_dir / f"predictions_{timestamp}.json"
            predictions_data = {
                'predictions': results['detailed_predictions']['predictions'],
                'targets': results['detailed_predictions']['targets'],
                'accuracy': results['metrics']['accuracy']
            }
            with open(pred_file, 'w') as f:
                json.dump(predictions_data, f, indent=2)
            print(f"Predictions saved to: {pred_file}")

        # Print comprehensive metrics
        print(f"\n{'='*60}")
        print(f"EVALUATION SUMMARY")
        print(f"{'='*60}")
        metrics_calc.print_metrics()

        # Print target achievement
        target_accuracy = 0.80  # 80% target from requirements
        achieved = results['metrics']['accuracy'] >= target_accuracy
        achievement_status = "✅ ACHIEVED" if achieved else "❌ NOT ACHIEVED"

        print(f"\n{'='*60}")
        print(f"TARGET ACCURACY ASSESSMENT")
        print(f"{'='*60}")
        print(f"Target Accuracy: {target_accuracy*100:.1f}%")
        print(f"Achieved Accuracy: {results['metrics']['accuracy']*100:.2f}%")
        print(f"Status: {achievement_status}")

        return results_file

    def run_evaluation(self):
        """Run complete evaluation pipeline"""
        print("Starting model evaluation...")

        # Load model
        model = self.load_model()

        # Create data loader
        test_loader = self.create_data_loader()

        # Evaluate model
        results, metrics_calc = self.evaluate(model, test_loader)

        # Save results
        results_file = self.save_results(results, metrics_calc)

        print(f"\nEvaluation completed successfully!")
        print(f"Results saved to: {results_file}")

        return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Evaluate trained CNN model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default='models/best.pth',
                        help='Path to model checkpoint')
    args = parser.parse_args()

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        print("Please train a model first or provide a valid checkpoint path.")
        return

    # Create evaluator and run evaluation
    evaluator = Evaluator(args.config, args.checkpoint)
    results = evaluator.run_evaluation()

    # Print final summary
    accuracy = results['metrics']['accuracy'] * 100
    print(f"\n🎯 Final Test Accuracy: {accuracy:.2f}%")

    if accuracy >= 80.0:
        print("🎉 Target accuracy (>80%) achieved!")
    else:
        print(f"📈 Need {80.0 - accuracy:.2f}% more to reach target accuracy")


if __name__ == '__main__':
    main()