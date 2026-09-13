"""Robust, two-sided null-calibrated evidence gate (not a probability)."""
from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class DecisionGate:
    center: float
    scale: float
    threshold: float
    false_fire_rate: float = .01
    null_windows: int = 0

    def __post_init__(self):
        if (not np.all(np.isfinite([self.center, self.scale, self.threshold]))
                or self.scale <= 0 or self.threshold <= 0
                or not 0 < self.false_fire_rate < .5):
            raise ValueError('Invalid calibrated gate.')

    @classmethod
    def fit(cls, differences, false_fire_rate=.01):
        values = np.asarray(differences, float)
        if values.ndim != 1 or len(values) < 20 or not np.all(np.isfinite(values)):
            raise ValueError('Calibration requires at least 20 finite null windows.')
        center = float(np.median(values))
        scale = float(1.4826 * np.median(np.abs(values - center)))
        if scale <= 1e-12:
            raise ValueError('Degenerate null calibration; collect more varied data.')
        threshold = float(np.quantile(np.abs((values-center)/scale),
                                      1-false_fire_rate, method='higher'))
        return cls(center, scale, threshold, false_fire_rate, len(values))

    def statistic(self, scores):
        return (float(scores[0]-scores[1]) - self.center) / self.scale

    def choice(self, scores):
        if scores is None or np.shape(scores) != (2,) or not np.all(np.isfinite(scores)):
            return None
        z = self.statistic(scores)
        return int(z < 0) if abs(z) > self.threshold else None

    def to_dict(self):
        return asdict(self)

    def metrics(self, scores, labels):
        choices = [self.choice(s) for s in scores]
        fired = [(c, y) for c, y in zip(choices, labels) if c is not None]
        return {'windows': len(choices), 'fires': len(fired),
                'fire_rate': len(fired)/len(choices) if choices else 0.,
                'precision': sum(c == y for c, y in fired)/len(fired) if fired else None}
