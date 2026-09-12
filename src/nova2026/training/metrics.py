"""Validation metric container used by
:class:`nova2026.training.trainer.SupervisedTrainer`.

A ``Metric`` bundles the scoring function with the direction that improves it,
so the trainer can select the best epoch without hard-coding F1 (classification)
or MAE (regression).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Metric:
    """A scoring function plus the direction that improves it.

    Parameters
    ----------
    fn
        ``fn(targets, predictions) -> float``, called with numpy arrays.
    greater_is_better
        ``True`` for accuracy / F1 / Spearman, ``False`` for MAE / MSE / loss.
        Prefer :meth:`maximize` / :meth:`minimize` to avoid getting this wrong.
    name
        Human-readable label used in logs and error messages.
    ravel
        Flatten both arrays with :func:`numpy.ravel` before scoring (the
        classification/regression default). Set ``False`` for metrics that need
        the original shape, such as multi-label, multi-output or segmentation
        scores; in that case the two arrays must have identical shapes.
    needs_extras
        ``True`` for metrics that also consume per-sample metadata (subject id,
        trial index, ...). Build those with :meth:`with_extras` so the trainer
        knows to gather the extras before scoring.

    Examples
    --------
    >>> from sklearn.metrics import f1_score, mean_absolute_error
    >>> macro_f1 = Metric.maximize(
    ...     lambda yt, yp: f1_score(yt, yp, average="macro"), name="macro_f1"
    ... )
    >>> mae = Metric.minimize(mean_absolute_error, name="mae")
    """

    fn: Callable[..., float]
    greater_is_better: bool = True
    name: str = "score"
    ravel: bool = True
    needs_extras: bool = False

    def __call__(self, targets, predictions, extras: Any = None) -> float:
        targets = np.asarray(targets)
        predictions = np.asarray(predictions)
        if self.ravel:
            targets, predictions = np.ravel(targets), np.ravel(predictions)
        elif targets.shape != predictions.shape:
            raise ValueError(
                f"Metric {self.name!r} requires aligned shapes when ravel=False; "
                f"got targets {targets.shape} and predictions {predictions.shape}."
            )
        if self.needs_extras:
            return float(self.fn(targets, predictions, extras))
        return float(self.fn(targets, predictions))

    def is_better(self, candidate: float | None, best: float | None) -> bool:
        """Whether ``candidate`` improves on ``best``.

        A missing or non-finite candidate never wins, so a NaN validation score
        (e.g. Spearman on a constant split) cannot be selected as the best epoch.
        """
        if candidate is None or not np.isfinite(candidate):
            return False
        if best is None:
            return True
        return candidate > best if self.greater_is_better else candidate < best

    def get_worse(self) -> float:
        """The worst possible value for this metric's direction."""
        return -float("inf") if self.greater_is_better else float("inf")

    @classmethod
    def maximize(
        cls,
        fn: Callable[[np.ndarray, np.ndarray], float],
        *,
        name: str = "score",
        ravel: bool = True,
    ) -> "Metric":
        """Metric where a larger value is better (accuracy, F1, rho, ...)."""
        return cls(fn=fn, greater_is_better=True, name=name, ravel=ravel)

    @classmethod
    def minimize(
        cls,
        fn: Callable[[np.ndarray, np.ndarray], float],
        *,
        name: str = "score",
        ravel: bool = True,
    ) -> "Metric":
        """Metric where a smaller value is better (MAE, MSE, loss, ...)."""
        return cls(fn=fn, greater_is_better=False, name=name, ravel=ravel)

    @classmethod
    def with_extras(
        cls,
        fn: Callable[[np.ndarray, np.ndarray, Any], float],
        *,
        greater_is_better: bool = True,
        name: str = "score",
        ravel: bool = True,
    ) -> "Metric":
        """Metric that also receives per-sample metadata.

        ``fn(targets, predictions, extras)``, where ``extras`` is the aligned
        output of the trainer's ``collect`` hook (subject ids, trial indices,
        ...). Grouping therefore never has to rely on loader ordering.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import spearmanr
        >>> def per_subject_rho(yt, yp, subjects):
        ...     return float(np.mean([
        ...         spearmanr(yt[subjects == s], yp[subjects == s]).statistic
        ...         for s in np.unique(subjects)
        ...     ]))
        >>> metric = Metric.with_extras(per_subject_rho, name="subject_rho")
        """
        return cls(
            fn=fn,
            greater_is_better=greater_is_better,
            name=name,
            ravel=ravel,
            needs_extras=True,
        )
