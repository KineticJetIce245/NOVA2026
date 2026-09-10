"""Ridge stimulus reconstruction with explicit post-stimulus EEG lags."""

import json

import numpy as np

from .config import AuditoryConfig


class RidgeDecoder:
    """Fit on aligned windows; keep all feature state fixed during inference."""

    def __init__(self, config=None, alpha=100.0):
        self.config = config if config is not None else AuditoryConfig()
        self.alpha = float(alpha)
        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("Ridge regularization must be positive.")
        self.weights = None
        self.mean = None
        self.scale = None
        self.contract = None
        self.training_info = {}

    def design(self, eeg):
        """Row t uses EEG[t:t+lag+1]; unavailable trailing rows are excluded."""
        count = len(eeg) - self.config.lag_samples
        if count < 2:
            raise ValueError("Window is too short for the configured lags.")
        normalized = (eeg - self.mean) / self.scale
        columns = []
        for lag in range(self.config.lag_samples + 1):
            columns.append(normalized[lag : lag + count])
        return np.concatenate(columns, axis=1)

    def fit(self, examples, training_info=None):
        """Examples are (AuditoryWindow, per-sample candidate labels)."""
        examples = list(examples)
        usable = []
        for window, labels in examples:
            if window.valid:
                usable.append((window, np.asarray(labels)))
        if not usable:
            raise ValueError("No valid training windows.")
        self.contract = usable[0][0].contract
        total = 0
        sums = np.zeros(usable[0][0].eeg.shape[1])
        squares = np.zeros_like(sums)
        for window, labels in usable:
            self.validate(window)
            if len(labels) != len(window.eeg):
                raise ValueError("Every training sample requires a target label.")
            total += len(window.eeg)
            sums += window.eeg.sum(axis=0)
            squares += (window.eeg * window.eeg).sum(axis=0)
        self.mean = sums / total
        variance = np.maximum(squares / total - self.mean * self.mean, 0)
        self.scale = np.sqrt(variance)
        if np.any(self.scale < 1e-9):
            raise ValueError("Training contains a flat EEG channel.")
        width = len(self.mean) * (self.config.lag_samples + 1)
        covariance = np.zeros((width, width))
        target_covariance = np.zeros(width)
        target_count = 0
        for window, labels in usable:
            design = self.design(window.eeg)
            labels = labels[: len(design)]
            selected = np.isin(labels, [0, 1])
            if not np.any(selected):
                continue
            target = window.envelopes[: len(design)][selected, labels[selected]]
            target = target - target.mean()
            target_scale = target.std()
            if target_scale < 1e-12:
                continue
            target = target / target_scale
            features = design[selected]
            covariance += features.T @ features
            target_covariance += features.T @ target
            target_count += len(target)
        if target_count == 0:
            raise ValueError("No varying labeled speech targets.")
        regularized = covariance + self.alpha * np.eye(width)
        self.weights = np.linalg.solve(regularized, target_covariance)
        self.training_info = dict(training_info or {})
        return self

    def validate(self, window):
        if window.contract != self.contract:
            raise ValueError("EEG preprocessing contract does not match the model.")
        if (
            window.contract is not None
            and "output_sfreq" in window.contract
            and window.contract["output_sfreq"] != self.config.sample_rate
        ):
            raise ValueError("EEG rate differs from the auditory feature rate.")
        if window.eeg.ndim != 2 or window.envelopes.shape != (len(window.eeg), 2):
            raise ValueError("Invalid aligned window dimensions.")
        if len(window.timestamps) != len(window.eeg):
            raise ValueError("Window timestamps do not match samples.")
        intervals = np.diff(window.timestamps)
        if not np.all(np.isfinite(window.timestamps)) or not np.allclose(
            intervals, 1 / self.config.sample_rate, atol=1e-6, rtol=0
        ):
            raise ValueError("Auditory window must use a continuous common time grid.")
        if not np.all(np.isfinite(window.eeg)) or not np.all(
            np.isfinite(window.envelopes)
        ):
            raise ValueError("Decoder inputs must be finite.")

    def score(self, window):
        if self.weights is None:
            raise ValueError("Decoder has not been trained.")
        self.validate(window)
        reconstruction = self.design(window.eeg) @ self.weights
        reconstruction = reconstruction - reconstruction.mean()
        scores = np.zeros(2)
        for candidate in range(2):
            envelope = window.envelopes[: len(reconstruction), candidate]
            envelope = envelope - envelope.mean()
            denominator = np.linalg.norm(reconstruction) * np.linalg.norm(envelope)
            if denominator > 1e-12:
                scores[candidate] = np.dot(reconstruction, envelope) / denominator
        return scores

    def save(self, path):
        if self.weights is None or self.mean is None or self.scale is None:
            raise ValueError("Cannot save an unfitted model.")
        metadata = {
            "version": 1,
            "features": self.config.to_dict(),
            "alpha": self.alpha,
            "contract": self.contract,
            "training_info": self.training_info,
        }
        np.savez_compressed(
            path,
            weights=self.weights,
            mean=self.mean,
            scale=self.scale,
            metadata=json.dumps(metadata),
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            if metadata["version"] != 1:
                raise ValueError("Unsupported model version.")
            features = dict(metadata["features"])
            method = features.pop("envelope_method")
            config = AuditoryConfig(**features)
            if method != config.to_dict()["envelope_method"]:
                raise ValueError("Unsupported envelope method.")
            model = cls(config, metadata["alpha"])
            model.contract = metadata["contract"]
            model.training_info = metadata["training_info"]
            model.weights = archive["weights"].copy()
            model.mean = archive["mean"].copy()
            model.scale = archive["scale"].copy()
        width = len(model.mean) * (config.lag_samples + 1)
        if model.weights.shape != (width,) or model.scale.shape != model.mean.shape:
            raise ValueError("Model array dimensions are inconsistent.")
        for array in (model.weights, model.mean, model.scale):
            if not np.all(np.isfinite(array)):
                raise ValueError("Model arrays must be finite.")
        if np.any(model.scale <= 0):
            raise ValueError("Model scales must be positive.")
        return model
