"""A saved, fixed EEG spatial projector with an explicit processing contract."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .config import StreamConfig
from .window import EEGWindow


def canonical_contract(config: StreamConfig) -> dict:
    """Normalize ordered tuples to their JSON representation for comparison."""

    return json.loads(json.dumps(config.processing_contract()))


class SpatialOperator:
    """Apply one EEG-only orthogonal projector after causal preprocessing.

    Args:
        matrix: Square EEG-channel projector in canonical order.
        contract: Exact channel, reference, and preprocessing requirements.
        report: Calibration measurements and provenance to retain with the matrix.

    Notes:
        EOG is never projected. Rank reduction is intentional; future covariance
        classifiers must regularize covariance matrices after projection.
    """

    def __init__(self, matrix: np.ndarray, contract: dict, report: dict) -> None:
        """Check projector dimensions, finiteness, symmetry, and idempotence."""

        matrix = np.asarray(matrix, dtype=np.float64).copy()
        count = len(contract["eeg_channels"])

        if matrix.shape != (count, count) or not np.all(np.isfinite(matrix)):
            raise ValueError("Projector shape and values do not match EEG channels.")

        if not np.allclose(matrix, matrix.T, atol=1e-8):
            raise ValueError("An orthogonal projector must be symmetric.")

        if not np.allclose(matrix @ matrix, matrix, atol=1e-8):
            raise ValueError("The matrix is not an idempotent projector.")
        rank = np.linalg.matrix_rank(matrix, tol=1e-7)

        if not 0 < rank < count:
            raise ValueError(
                "The projector must remove some, but not all, EEG directions."
            )

        self.matrix = matrix
        self.matrix.setflags(write=False)
        self.contract = json.loads(json.dumps(contract))
        self.report = json.loads(json.dumps(report))
        encoded = json.dumps(self.contract, sort_keys=True).encode()
        self.artifact_id = hashlib.sha256(matrix.tobytes() + encoded).hexdigest()

    def validate_config(self, config: StreamConfig) -> None:
        """Reject reordered channels or different reference/filter/rate settings."""

        if canonical_contract(config) != self.contract:
            raise ValueError(
                "Artifact operator and run preprocessing contracts differ."
            )

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Return projected samples-by-EEG-channel data without changing input."""

        if data.ndim != 2 or data.shape[1] != self.matrix.shape[0]:
            raise ValueError("Expected samples by the saved EEG channel count.")

        if not np.all(np.isfinite(data)):
            raise ValueError("Do not project nonfinite data.")

        return data @ self.matrix.T

    def apply_window(self, window: EEGWindow) -> EEGWindow:
        """Project a window exactly once, preserving EOG, timing, and validity."""

        if window.artifact_id is not None:
            raise ValueError("This window already has a spatial correction.")

        return EEGWindow(
            self.apply(window.data),
            window.eog.copy(),
            window.timestamps.copy(),
            window.valid,
            window.reasons,
            window.start_sample,
            window.segment,
            self.artifact_id,
            window.channel_names,
            window.contract,
            window.interpolated_samples,
            window.interpolated,
            window.available_at,
        )

    def save(self, path: Path) -> None:
        """Save matrix and metadata together, refusing to overwrite an operator."""

        metadata = json.dumps(
            {
                "schema": 1,
                "contract": self.contract,
                "report": self.report,
                "artifact_id": self.artifact_id,
            },
            allow_nan=False,
        )
        with Path(path).open("xb") as stream:
            np.savez_compressed(stream, matrix=self.matrix, metadata=np.array(metadata))

    @classmethod
    def load(cls, path: Path) -> "SpatialOperator":
        """Load numeric data without pickle and verify its content identifier."""

        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            if metadata["schema"] != 1:
                raise ValueError("Unsupported artifact format.")
            operator = cls(data["matrix"], metadata["contract"], metadata["report"])

        if operator.artifact_id != metadata["artifact_id"]:
            raise ValueError(
                "The saved artifact identifier does not match its content."
            )

        return operator
