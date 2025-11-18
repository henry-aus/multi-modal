"""
Naive Bayes classifier for text classification.
"""
from typing import Any
from sklearn.naive_bayes import MultinomialNB
from .base import BaseClassicalModel


class NaiveBayesClassifier(BaseClassicalModel):
    """
    Naive Bayes classifier using scikit-learn's MultinomialNB.

    Particularly effective for text classification with TF-IDF features.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_prior: bool = True,
        class_prior: Any = None,
        **kwargs
    ):
        """
        Initialize NaiveBayesClassifier.

        Args:
            alpha (float): Additive smoothing parameter
            fit_prior (bool): Whether to learn class prior probabilities
            class_prior (array-like, optional): Prior probabilities of the classes
            **kwargs: Additional arguments for BaseClassicalModel
        """
        super().__init__(**kwargs)

        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior

    def _create_model(self) -> MultinomialNB:
        """
        Create and return the Naive Bayes model instance.

        Returns:
            MultinomialNB: Sklearn Naive Bayes model
        """
        return MultinomialNB(
            alpha=self.alpha,
            fit_prior=self.fit_prior,
            class_prior=self.class_prior
        )