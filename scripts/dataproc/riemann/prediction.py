"""Timestamped classification results, including explicit rejected-window results."""

import numpy as np
from mne_lsl.lsl import local_clock

from scripts.dataproc.streaming.window import EEGWindow


class Prediction:
    """Carry class probabilities with source time, delivery time and validity.

    Args:
        window: Original live window, including repair/segment information.
        model_id: Identifier of the fitted model used for this result.
        classes: Class labels in probability-column order.
        probabilities: One probability per class, or None for a rejected window.
    """

    def __init__(
        self,
        window: EEGWindow,
        model_id: str,
        classes: np.ndarray,
        probabilities: np.ndarray | None,
    ) -> None:
        """Retain timing and quality; never manufacture probabilities for rejection."""

        if window.valid != (probabilities is not None):
            raise ValueError("Only valid windows can carry probabilities.")
        if probabilities is not None:
            if probabilities.shape != (len(classes),) or not np.all(
                np.isfinite(probabilities)
            ):
                raise ValueError("Invalid probability dimensions or values.")
            if (
                np.any(probabilities < 0)
                or np.any(probabilities > 1)
                or not np.isclose(probabilities.sum(), 1)
            ):
                raise ValueError(
                    "Class probabilities must be between zero and one and sum to one."
                )

        self.start_time = float(window.timestamps[0])
        self.end_time = float(window.timestamps[-1])
        self.available_at = window.available_at
        self.emitted_at = local_clock()
        self.segment = window.segment
        self.start_sample = window.start_sample
        self.valid = window.valid
        self.reasons = window.reasons
        self.interpolated = window.interpolated
        self.interpolated_samples = window.interpolated_samples
        self.model_id = model_id
        self.classes = classes.tolist()
        self.probabilities = None if probabilities is None else probabilities.tolist()

    def to_dict(self) -> dict:
        """Return a JSON-compatible result for the controller or recorder."""

        return dict(vars(self))
