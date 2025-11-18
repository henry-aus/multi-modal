"""
Base class for classical ML models.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import joblib
import logging

logger = logging.getLogger(__name__)


class BaseClassicalModel(ABC):
    """
    Base class for classical ML text classification models.
    """

    def __init__(
        self,
        vectorizer_type: str = "tfidf",
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
        random_state: int = 42
    ):
        """
        Initialize BaseClassicalModel.

        Args:
            vectorizer_type (str): Type of vectorizer ("tfidf" or "count")
            max_features (int): Maximum number of features
            ngram_range (Tuple[int, int]): N-gram range for feature extraction
            random_state (int): Random state for reproducibility
        """
        self.vectorizer_type = vectorizer_type
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.random_state = random_state

        # Initialize vectorizer
        if vectorizer_type == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words='english',
                lowercase=True,
                strip_accents='unicode'
            )
        elif vectorizer_type == "count":
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words='english',
                lowercase=True,
                strip_accents='unicode'
            )
        else:
            raise ValueError(f"Unsupported vectorizer type: {vectorizer_type}")

        # Model will be initialized by subclass
        self.model = None
        self.is_fitted = False

        # Metadata
        self.feature_names = None
        self.num_classes = None
        self.class_labels = None

        logger.info(f"Initialized {self.__class__.__name__} with {vectorizer_type} vectorizer")

    @abstractmethod
    def _create_model(self) -> Any:
        """
        Create and return the sklearn model instance.

        Returns:
            Any: Sklearn model instance
        """
        pass

    def fit(self, texts: List[str], labels: List[str]) -> 'BaseClassicalModel':
        """
        Fit the model on training data.

        Args:
            texts (List[str]): Training texts
            labels (List[str]): Training labels

        Returns:
            BaseClassicalModel: Self
        """
        logger.info(f"Fitting {self.__class__.__name__} on {len(texts)} samples")

        # Vectorize texts
        X = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()

        # Prepare labels
        y = np.array(labels)
        self.class_labels = np.unique(y)
        self.num_classes = len(self.class_labels)

        # Create and fit model
        self.model = self._create_model()
        self.model.fit(X, y)

        self.is_fitted = True

        logger.info(f"Model fitted successfully")
        logger.info(f"  Feature matrix shape: {X.shape}")
        logger.info(f"  Number of classes: {self.num_classes}")
        logger.info(f"  Classes: {self.class_labels}")

        return self

    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Make predictions on new texts.

        Args:
            texts (List[str]): Texts to predict

        Returns:
            np.ndarray: Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        X = self.vectorizer.transform(texts)
        predictions = self.model.predict(X)

        return predictions

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            texts (List[str]): Texts to predict

        Returns:
            np.ndarray: Prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model does not support probability predictions")

        X = self.vectorizer.transform(texts)
        probabilities = self.model.predict_proba(X)

        return probabilities

    def evaluate(self, texts: List[str], labels: List[str]) -> Dict[str, Any]:
        """
        Evaluate model performance.

        Args:
            texts (List[str]): Test texts
            labels (List[str]): True labels

        Returns:
            Dict[str, Any]: Evaluation metrics
        """
        predictions = self.predict(texts)
        y_true = np.array(labels)

        # Calculate metrics
        accuracy = accuracy_score(y_true, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, predictions, average=None, labels=self.class_labels
        )

        # Macro and weighted averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, predictions, average='macro'
        )
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            y_true, predictions, average='weighted'
        )

        # Confusion matrix
        conf_matrix = confusion_matrix(y_true, predictions, labels=self.class_labels)

        metrics = {
            'accuracy': accuracy,
            'precision_per_class': precision.tolist(),
            'recall_per_class': recall.tolist(),
            'f1_per_class': f1.tolist(),
            'support_per_class': support.tolist(),
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
            'confusion_matrix': conf_matrix.tolist(),
            'class_labels': self.class_labels.tolist()
        }

        return metrics

    def cross_validate(
        self,
        texts: List[str],
        labels: List[str],
        cv: int = 5,
        scoring: str = 'accuracy'
    ) -> Dict[str, Any]:
        """
        Perform cross-validation.

        Args:
            texts (List[str]): Texts for cross-validation
            labels (List[str]): Labels for cross-validation
            cv (int): Number of cross-validation folds
            scoring (str): Scoring metric

        Returns:
            Dict[str, Any]: Cross-validation results
        """
        # Vectorize texts
        X = self.vectorizer.fit_transform(texts)
        y = np.array(labels)

        # Create fresh model for cross-validation
        model = self._create_model()

        # Perform cross-validation
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

        results = {
            'scores': scores.tolist(),
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'min_score': scores.min(),
            'max_score': scores.max(),
            'scoring_metric': scoring,
            'cv_folds': cv
        }

        logger.info(f"Cross-validation results ({scoring}):")
        logger.info(f"  Mean: {results['mean_score']:.4f} (+/- {results['std_score'] * 2:.4f})")
        logger.info(f"  Min: {results['min_score']:.4f}, Max: {results['max_score']:.4f}")

        return results

    def get_feature_importance(self, top_n: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get feature importance for each class (if supported by the model).

        Args:
            top_n (int): Number of top features to return per class

        Returns:
            Dict[str, List[Tuple[str, float]]]: Feature importance per class
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")

        if not hasattr(self.model, 'coef_'):
            logger.warning("Model does not support feature importance extraction")
            return {}

        importance_dict = {}

        # For binary classification, coef_ has shape (1, n_features)
        # For multi-class, coef_ has shape (n_classes, n_features)
        coef = self.model.coef_

        if coef.shape[0] == 1:  # Binary classification
            # Get positive and negative features
            feature_scores = [(self.feature_names[i], coef[0, i]) for i in range(len(self.feature_names))]

            # Sort by absolute importance
            feature_scores_pos = sorted(
                [(name, score) for name, score in feature_scores if score > 0],
                key=lambda x: abs(x[1]), reverse=True
            )[:top_n]

            feature_scores_neg = sorted(
                [(name, score) for name, score in feature_scores if score < 0],
                key=lambda x: abs(x[1]), reverse=True
            )[:top_n]

            importance_dict['positive_class'] = feature_scores_pos
            importance_dict['negative_class'] = feature_scores_neg

        else:  # Multi-class classification
            for i, class_label in enumerate(self.class_labels):
                feature_scores = [
                    (self.feature_names[j], coef[i, j])
                    for j in range(len(self.feature_names))
                ]

                # Sort by importance (descending)
                feature_scores = sorted(feature_scores, key=lambda x: x[1], reverse=True)
                importance_dict[str(class_label)] = feature_scores[:top_n]

        return importance_dict

    def save_model(self, filepath: str):
        """
        Save the trained model to disk.

        Args:
            filepath (str): Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'vectorizer_type': self.vectorizer_type,
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'random_state': self.random_state,
            'feature_names': self.feature_names,
            'num_classes': self.num_classes,
            'class_labels': self.class_labels,
            'is_fitted': self.is_fitted
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """
        Load a trained model from disk.

        Args:
            filepath (str): Path to the saved model
        """
        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.vectorizer_type = model_data['vectorizer_type']
        self.max_features = model_data['max_features']
        self.ngram_range = model_data['ngram_range']
        self.random_state = model_data['random_state']
        self.feature_names = model_data['feature_names']
        self.num_classes = model_data['num_classes']
        self.class_labels = model_data['class_labels']
        self.is_fitted = model_data['is_fitted']

        logger.info(f"Model loaded from {filepath}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.

        Returns:
            Dict[str, Any]: Model information
        """
        info = {
            'model_type': self.__class__.__name__,
            'vectorizer_type': self.vectorizer_type,
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'random_state': self.random_state,
            'is_fitted': self.is_fitted
        }

        if self.is_fitted:
            info.update({
                'num_classes': self.num_classes,
                'class_labels': self.class_labels.tolist() if self.class_labels is not None else None,
                'num_features': len(self.feature_names) if self.feature_names is not None else None
            })

        return info