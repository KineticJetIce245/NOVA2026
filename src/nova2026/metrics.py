"""Validation metric container used by :class:`nova2026.trainer.SupervisedTrainer`.

A ``Metric`` bundles the scoring function with the direction that improves it,
so the trainer can select the best epoch without hard-coding F1 (classification)
or MAE (regression).
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Metric:
    fn: Callable[[np.ndarray, np.ndarray], float]
    greater_is_better: bool = True
    name: str = "score"

    def __call__(self, targets, predictions) -> float:
        return float(self.fn(np.ravel(targets), np.ravel(predictions)))

    def is_better(self, candidate: float, best: float | None) -> bool:
        if best is None:
            return True
        return candidate > best if self.greater_is_better else candidate < best

    def get_worse(self) -> float:
        return -float("inf") if self.greater_is_better else float("inf")
