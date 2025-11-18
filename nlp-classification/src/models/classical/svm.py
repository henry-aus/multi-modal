"""
Support Vector Machine classifier for text classification.
"""
from typing import Any, Union
from sklearn.svm import SVC
from .base import BaseClassicalModel


class SVMClassifier(BaseClassicalModel):
    """
    Support Vector Machine classifier using scikit-learn's SVC.

    Effective for high-dimensional sparse data like text features.
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = 'rbf',
        degree: int = 3,
        gamma: Union[str, float] = 'scale',
        coef0: float = 0.0,
        shrinking: bool = True,
        probability: bool = True,
        tol: float = 1e-3,
        cache_size: float = 200,
        class_weight: Any = None,
        verbose: bool = False,
        max_iter: int = -1,
        decision_function_shape: str = 'ovr',
        break_ties: bool = False,
        **kwargs
    ):
        """
        Initialize SVMClassifier.

        Args:
            C (float): Regularization parameter
            kernel (str): Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
            degree (int): Degree of polynomial kernel
            gamma (Union[str, float]): Kernel coefficient for 'rbf', 'poly', 'sigmoid'
            coef0 (float): Independent term in kernel function
            shrinking (bool): Whether to use shrinking heuristic
            probability (bool): Whether to enable probability estimates
            tol (float): Tolerance for stopping criterion
            cache_size (float): Size of kernel cache (in MB)
            class_weight (Any): Weights associated with classes
            verbose (bool): Enable verbose output
            max_iter (int): Maximum number of iterations (-1 for no limit)
            decision_function_shape (str): Decision function shape ('ovo', 'ovr')
            break_ties (bool): Break ties according to confidence values
            **kwargs: Additional arguments for BaseClassicalModel
        """
        super().__init__(**kwargs)

        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.probability = probability
        self.tol = tol
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.verbose = verbose
        self.max_iter = max_iter
        self.decision_function_shape = decision_function_shape
        self.break_ties = break_ties

    def _create_model(self) -> SVC:
        """
        Create and return the SVM model instance.

        Returns:
            SVC: Sklearn SVM model
        """
        return SVC(
            C=self.C,
            kernel=self.kernel,
            degree=self.degree,
            gamma=self.gamma,
            coef0=self.coef0,
            shrinking=self.shrinking,
            probability=self.probability,
            tol=self.tol,
            cache_size=self.cache_size,
            class_weight=self.class_weight,
            verbose=self.verbose,
            max_iter=self.max_iter,
            decision_function_shape=self.decision_function_shape,
            break_ties=self.break_ties,
            random_state=self.random_state
        )