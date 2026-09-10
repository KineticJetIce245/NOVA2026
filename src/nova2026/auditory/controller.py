"""Three-state control with evidence expiry independent of inference."""

import math

import numpy as np


class AttentionController:
    """Own this component on the audio thread; hand estimates to it via a queue."""

    def __init__(self, margin=0.03, max_age=3.0, attenuation_db=6.0):
        if not all(math.isfinite(value) for value in (margin, max_age, attenuation_db)):
            raise ValueError("Controller parameters must be finite.")
        if margin <= 0 or max_age <= 0 or not 0 <= attenuation_db <= 20:
            raise ValueError("Invalid controller thresholds.")
        self.margin = margin
        self.max_age = max_age
        self.duck = 10 ** (-attenuation_db / 20)
        self.selected = None
        self.evidence_end = -math.inf
        self.manual = None
        self.manual_enabled = False

    def set_manual(self, enabled, candidate=None):
        if candidate not in (None, 0, 1):
            raise ValueError("Manual candidate must be A, B, or neutral.")
        self.manual_enabled = bool(enabled)
        self.manual = candidate

    def update(self, estimate, now):
        scores = estimate.scores
        if (
            not estimate.valid
            or scores is None
            or scores.shape != (2,)
            or not np.all(np.isfinite(scores))
            or not math.isfinite(estimate.evidence_end)
            or not math.isfinite(estimate.emitted_at)
        ):
            self.selected = None
            return
        if estimate.evidence_end < self.evidence_end:
            return
        age = now - estimate.evidence_end
        if age < 0 or age > self.max_age or estimate.emitted_at > now:
            self.selected = None
            return
        self.evidence_end = estimate.evidence_end
        difference = scores[0] - scores[1]
        if abs(difference) < self.margin:
            self.selected = None
        elif difference > 0:
            self.selected = 0
        else:
            self.selected = 1

    def choice(self, now):
        if self.manual_enabled:
            return self.manual
        if now - self.evidence_end > self.max_age:
            return None
        return self.selected

    def gains(self, now):
        gains = np.ones(2)
        selected = self.choice(now)
        if selected is not None:
            gains[1 - selected] = self.duck
        return gains
