"""Ridge regression readout layer for quantum reservoir computing.

This module provides a ridge regression implementation using the closed-form
solution for mapping quantum reservoir features to predictions.
"""

import numpy as np


class RidgeReadout:
    """Ridge regression readout layer with closed-form solution.

    Uses Tikhonov regularization to compute optimal weights via the closed-form
    solution: W = (XᵀX + αI)⁻¹Xᵀy, where α is the regularization parameter.

    Attributes:
        alpha: Regularization parameter (default: 1e-4).
        _weights: Learned weight matrix, None until fit() is called.

    Example:
        >>> readout = RidgeReadout(alpha=1e-3)
        >>> readout.fit(train_features, train_targets)
        >>> predictions = readout.predict(test_features)
    """

    def __init__(self, alpha: float = 1e-4):
        """Initialize ridge regression readout.

        Args:
            alpha: Regularization parameter for ridge regression. Higher values
                increase regularization strength. Default: 1e-4.
        """
        self.alpha = alpha
        self._weights: np.ndarray | None = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "RidgeReadout":
        """Fit ridge regression weights using closed-form solution.

        Computes W = (XᵀX + αI)⁻¹Xᵀy where:
        - X is the feature matrix (N samples × d features)
        - y is the target vector (N samples)
        - α is the regularization parameter
        - I is the identity matrix

        Args:
            features: Feature matrix of shape (N, d) where N is the number of
                samples and d is the feature dimension.
            targets: Target vector of shape (N,) or (N, h) where h is the
                forecast horizon.

        Returns:
            Self, for method chaining.

        Raises:
            ValueError: If the number of samples in features and targets don't match.
        """
        if features.shape[0] != targets.shape[0]:
            raise ValueError(
                f"Feature and target sample counts must match. "
                f"Got features: {features.shape[0]}, targets: {targets.shape[0]}"
            )

        X, y = features, targets
        # Compute (XᵀX + αI)⁻¹Xᵀy using np.linalg.solve for numerical stability
        self._weights = np.linalg.solve(
            X.T @ X + self.alpha * np.eye(X.shape[1]),
            X.T @ y,
        )
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Generate predictions using learned weights.

        Args:
            features: Feature matrix of shape (N, d) where N is the number of
                samples and d is the feature dimension (must match training).

        Returns:
            Predictions of shape (N,) or (N, h) depending on target shape during fit.

        Raises:
            RuntimeError: If predict() is called before fit().
        """
        if self._weights is None:
            raise RuntimeError("Must call fit() before predict()")
        result: np.ndarray = features @ self._weights
        return result
