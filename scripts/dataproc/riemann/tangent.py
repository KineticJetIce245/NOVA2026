"""Fixed-reference tangent features for symmetric positive-definite covariances."""

import numpy as np

from .covariance import symmetric_power


class TangentSpace:
    """Fit a log-Euclidean reference, then map covariances into tangent features.

    Notes:
        fit() belongs to the caller's training split only. transform() never
        updates the reference. The reference method is explicitly log-Euclidean;
        it is not an iterative affine-invariant mean.
    """

    def __init__(self) -> None:
        """Start unfitted; inference requires fit() or set_reference()."""

        self.reference: np.ndarray | None = None
        self.whitener: np.ndarray | None = None

    def _log(self, matrix: np.ndarray) -> np.ndarray:
        """Return the symmetric matrix logarithm after checking positive definiteness."""

        values, vectors = np.linalg.eigh(matrix)
        if np.any(values <= 0) or not np.all(np.isfinite(values)):
            raise ValueError("Tangent input must be positive definite.")
        return (vectors * np.log(values)) @ vectors.T

    def _validate(self, covariances: np.ndarray) -> None:
        """Reject empty, nonfinite, nonsquare or nonsymmetric covariance batches."""

        if (
            covariances.ndim != 3
            or not len(covariances)
            or covariances.shape[1] != covariances.shape[2]
        ):
            raise ValueError("Expected windows by channels by channels.")
        if not np.all(np.isfinite(covariances)) or not np.allclose(
            covariances, covariances.transpose(0, 2, 1)
        ):
            raise ValueError("Covariances must be finite and symmetric.")

    def fit(self, covariances: np.ndarray) -> "TangentSpace":
        """Fit the reference on training covariances only; return this component."""

        self._validate(covariances)
        mean_log = np.mean([self._log(matrix) for matrix in covariances], axis=0)
        values, vectors = np.linalg.eigh(mean_log)
        self.set_reference((vectors * np.exp(values)) @ vectors.T)
        return self

    def set_reference(self, reference: np.ndarray) -> None:
        """Load an already-fitted reference and compute its inverse square root."""

        reference = np.asarray(reference, dtype=np.float64)
        if reference.ndim != 2 or not len(reference):
            raise ValueError("A reference must be a non-empty square matrix.")
        self._validate(reference[None, :, :])
        self.whitener = symmetric_power(reference, -0.5)
        self.reference = reference.copy()

    def transform(self, covariances: np.ndarray) -> np.ndarray:
        """Return (windows, C*(C+1)/2) features at the fixed training reference."""

        self._validate(covariances)
        if self.reference is None or self.whitener is None:
            raise RuntimeError("Fit or load the tangent reference before inference.")
        if covariances.shape[1:] != self.reference.shape:
            raise ValueError("Covariance channels differ from the fitted reference.")
        rows, columns = np.triu_indices(len(self.reference))
        weights = np.where(rows == columns, 1.0, np.sqrt(2.0))
        result = []
        for covariance in covariances:
            logged = self._log(self.whitener @ covariance @ self.whitener)
            result.append(logged[rows, columns] * weights)
        return np.stack(result)
