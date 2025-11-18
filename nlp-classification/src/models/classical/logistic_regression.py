"""
Logistic Regression classifier for text classification.
"""
from typing import Any, Union
from sklearn.linear_model import LogisticRegression
from .base import BaseClassicalModel


class LogisticRegressionClassifier(BaseClassicalModel):
    """
    Logistic Regression classifier using scikit-learn's LogisticRegression.

    Fast and effective baseline for text classification tasks.
    """

    def __init__(
        self,
        penalty: str = 'l2',
        dual: bool = False,
        tol: float = 1e-4,
        C: float = 1.0,
        fit_intercept: bool = True,
        intercept_scaling: float = 1,
        class_weight: Any = None,
        solver: str = 'lbfgs',
        max_iter: int = 1000,
        multi_class: str = 'auto',
        verbose: int = 0,
        warm_start: bool = False,
        n_jobs: Any = None,
        l1_ratio: Any = None,
        **kwargs
    ):
        """
        Initialize LogisticRegressionClassifier.

        Args:
            penalty (str): Penalty norm ('l1', 'l2', 'elasticnet', 'none')
            dual (bool): Dual or primal formulation
            tol (float): Tolerance for stopping criteria
            C (float): Inverse of regularization strength
            fit_intercept (bool): Whether to fit intercept
            intercept_scaling (float): Scaling for synthetic feature
            class_weight (Any): Weights associated with classes
            solver (str): Algorithm for optimization ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga')
            max_iter (int): Maximum iterations for solvers
            multi_class (str): Multi-class strategy ('auto', 'ovr', 'multinomial')
            verbose (int): Verbosity level
            warm_start (bool): Reuse solution of previous call
            n_jobs (Any): Number of CPU cores for parallel computation
            l1_ratio (Any): Elastic-Net mixing parameter
            **kwargs: Additional arguments for BaseClassicalModel
        """
        super().__init__(**kwargs)

        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.l1_ratio = l1_ratio

    def _create_model(self) -> LogisticRegression:
        """
        Create and return the Logistic Regression model instance.

        Returns:
            LogisticRegression: Sklearn Logistic Regression model
        """
        return LogisticRegression(
            penalty=self.penalty,
            dual=self.dual,
            tol=self.tol,
            C=self.C,
            fit_intercept=self.fit_intercept,
            intercept_scaling=self.intercept_scaling,
            class_weight=self.class_weight,
            random_state=self.random_state,
            solver=self.solver,
            max_iter=self.max_iter,
            multi_class=self.multi_class,
            verbose=self.verbose,
            warm_start=self.warm_start,
            n_jobs=self.n_jobs,
            l1_ratio=self.l1_ratio
        )