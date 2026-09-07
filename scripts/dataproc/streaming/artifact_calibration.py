"""Estimate an EOG-guided signal-space projector from marked calibration epochs."""

import numpy as np

from .config import StreamConfig
from .spatial_operator import SpatialOperator, canonical_contract


class ArtifactCalibration:
    """Fit artifact directions on training blinks and inspect held-out events.

    Args:
        config: Same causal preprocessing and reference used during application.
        n_components: Number of EOG-associated spatial directions to remove.

    Notes:
        This is EOG-guided SSP, not ICA or sample interpolation. Cross-covariance
        identifies EEG directions coupled to measured EOG. Any brain activity
        sharing those directions is also attenuated. Inspect calibration traces
        and spatial weights before selecting the operator for human recordings.
    """

    def __init__(self, config: StreamConfig, n_components: int = 1) -> None:
        """Require recorded EOG and a small, explicit artifact subspace."""

        if not config.eog_channels:
            raise ValueError("EOG calibration requires actual recorded EOG channels.")
        maximum = min(len(config.eog_channels), len(config.eeg_channels) - 1)

        if isinstance(n_components, bool) or not isinstance(n_components, int):
            raise TypeError("The number of artifact components must be an integer.")

        if not 1 <= n_components <= maximum:
            raise ValueError("Unsupported number of artifact components.")
        self.config = config.updated()
        self.n_components = n_components

    def fit(self, eeg: np.ndarray, eog: np.ndarray) -> SpatialOperator:
        """Fit a fixed projector using non-overlapping, marked blink/movement epochs.

        Args:
            eeg: Events by samples by EEG channels, after causal preprocessing.
            eog: Matching measured EOG epochs in microvolts.

        Returns:
            A projector with held-out coupling, energy, and spatial-weight reports.

        Raises:
            ValueError: If data is insufficient, flat, nonfinite, or held-out
                EOG coupling is not reduced by at least half.
        """

        if eeg.ndim != 3 or eog.ndim != 3 or eeg.shape[:2] != eog.shape[:2]:
            raise ValueError(
                "EEG and EOG must contain matching event/sample dimensions."
            )

        if eeg.shape[2] != len(self.config.eeg_channels) or eog.shape[2] != len(
            self.config.eog_channels
        ):
            raise ValueError("Calibration channel counts do not match the contract.")

        if len(eeg) < 6 or eeg.shape[1] < self.config.output_sfreq / 2:
            raise ValueError(
                "Provide at least six non-overlapping events, each at least 0.5 seconds."
            )

        if not np.all(np.isfinite(eeg)) or not np.all(np.isfinite(eog)):
            raise ValueError("Calibration epochs must be finite.")

        # Keep complete events together. The last third is never used for fitting.
        split = len(eeg) - max(2, len(eeg) // 3)
        eeg = eeg - eeg.mean(axis=1, keepdims=True)
        eog = eog - eog.mean(axis=1, keepdims=True)
        train_eeg = eeg[:split].reshape(-1, eeg.shape[2])
        train_eog = eog[:split].reshape(-1, eog.shape[2])
        test_eeg = eeg[split:].reshape(-1, eeg.shape[2])
        test_eog = eog[split:].reshape(-1, eog.shape[2])

        if np.any(np.std(train_eog, axis=0) < 1.0) or np.any(
            np.std(test_eog, axis=0) < 1.0
        ):
            raise ValueError(
                "Training and held-out EOG must contain representative movement."
            )

        cross_covariance = train_eeg.T @ train_eog / len(train_eeg)
        directions, singular_values, _ = np.linalg.svd(
            cross_covariance, full_matrices=False
        )

        if singular_values[self.n_components - 1] < 1e-6:
            raise ValueError("No measurable EOG-associated EEG direction was found.")
        basis = directions[:, : self.n_components]
        matrix = np.eye(eeg.shape[2]) - basis @ basis.T
        corrected = test_eeg @ matrix.T

        before = float(np.linalg.norm(test_eeg.T @ test_eog))
        after = float(np.linalg.norm(corrected.T @ test_eog))

        if before < 1e-6 or after / before > 0.5:
            raise ValueError(
                "The fitted operator did not sufficiently reduce held-out EOG coupling."
            )
        energy = float(np.sum(test_eeg**2))
        report = {
            "method": "EOG-guided SSP via EEG/EOG cross-covariance",
            "training_events": split,
            "heldout_events": len(eeg) - split,
            "samples_per_event": eeg.shape[1],
            "n_components": self.n_components,
            "heldout_coupling_ratio": after / before,
            "heldout_energy_retained": float(np.sum(corrected**2)) / energy,
            "directions": basis.tolist(),
            "singular_values": singular_values.tolist(),
            "review_note": "Coupling reduction does not prove preservation of brain activity.",
        }

        return SpatialOperator(matrix, canonical_contract(self.config), report)
