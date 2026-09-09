"""Regularized covariance with an explicit positive-definite check."""

import numpy as np
from sklearn.covariance import ledoit_wolf


def symmetric_power(matrix: np.ndarray, power: float) -> np.ndarray:
    """Raise a symmetric positive-definite matrix to a real power."""

    values, vectors = np.linalg.eigh(matrix)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("A covariance/reference must be positive definite.")
    return (vectors * values**power) @ vectors.T


class CovarianceEstimator:
    """Estimate each window independently using Ledoit-Wolf shrinkage and a ridge.

    Args:
        ridge: Positive ridge as a fraction of mean channel variance. This keeps
            covariance invertible after a rank-reducing artifact projection.
    """

    def __init__(self, ridge: float = 1e-6) -> None:
        """Validate the fixed regularization strength; no dataset fitting occurs."""

        if not np.isfinite(ridge) or ridge <= 0:
            raise ValueError("Covariance ridge must be finite and positive.")
        self.ridge = ridge

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Return (windows, channels, channels) from (windows, samples, channels)."""

        if data.ndim != 3 or not len(data) or data.shape[1] < 2 or data.shape[2] < 1:
            raise ValueError("Expected non-empty windows by samples by channels.")
        if not np.all(np.isfinite(data)):
            raise ValueError("Covariance input must be finite.")
        result = []
        for window in data:
            covariance, _ = ledoit_wolf(window)
            variance = float(np.trace(covariance) / len(covariance))
            if not np.isfinite(variance) or variance <= 0:
                raise ValueError("A flat window has no usable covariance.")
            covariance += np.eye(len(covariance)) * variance * self.ridge
            np.linalg.cholesky(covariance)
            result.append(covariance)
        return np.stack(result)
