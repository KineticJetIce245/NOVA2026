"""Timestamped EEG windows passed between processing components."""

import numpy as np


class EEGWindow:
    """One output window with its timing and quality decision.

    Attributes:
        data: EEG samples by channels, in microvolts and canonical EEG order.
        eog: Separate auxiliary samples by channels, also in microvolts.
        timestamps: Nominal source-time grid at the configured output rate.
        valid: Whether the window may enter the downstream pipeline.
        reasons: Rejection reasons; empty for a valid window.
        start_sample: First sample index within this segment's output grid.
        segment: Contiguous processing segment; increases after recovery.
        artifact_id: Applied operator's content identifier, or None if uncorrected.
        channel_names: EEG order already selected by the acquisition contract.
        contract: Serializable processing/window settings for model compatibility.
        interpolated_samples: Source sample rows in repair spans touching this window.
        interpolated: Also marks the configured settling interval after a repair.
        available_at: Latest source time consumed for this output batch, including
            interpolation endpoints. Conservative when chunks contain several windows;
            actual delivery additionally includes queue and computation delay.
    """

    def __init__(
        self,
        data: np.ndarray,
        eog: np.ndarray,
        timestamps: np.ndarray,
        valid: bool,
        reasons: tuple[str, ...],
        start_sample: int,
        segment: int = 0,
        artifact_id: str | None = None,
        channel_names: tuple[str, ...] = (),
        contract: dict | None = None,
        interpolated_samples: int = 0,
        interpolated: bool = False,
        available_at: float | None = None,
    ) -> None:
        """Store one window and its explicit provenance and validity."""

        self.data = data
        self.eog = eog
        self.timestamps = timestamps
        self.valid = valid
        self.reasons = reasons
        self.start_sample = start_sample
        self.segment = segment
        self.artifact_id = artifact_id
        self.channel_names = tuple(channel_names)
        self.contract = contract
        self.interpolated_samples = interpolated_samples
        self.interpolated = interpolated
        self.available_at = available_at
