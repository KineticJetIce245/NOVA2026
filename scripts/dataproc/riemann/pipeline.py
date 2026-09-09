"""Live covariance/tangent/logistic inference using the project's Pipeline tubes."""

import numpy as np

from nova2026.data.pipeline import Pipeline
from scripts.dataproc.streaming.window import EEGWindow

from .adapter import WindowAdapter
from .model import RiemannModel
from .prediction import Prediction


class InferenceFrame:
    """Keep one window and its intermediate arrays together through the tubes."""

    def __init__(self, window: EEGWindow) -> None:
        """Start an empty per-call frame; no state is shared between windows."""

        self.window = window
        self.data: np.ndarray | None = None
        self.covariances: np.ndarray | None = None
        self.features: np.ndarray | None = None


class RiemannPipeline(Pipeline):
    """Validate, estimate covariance, map tangent features and predict in order.

    Args:
        model: Fitted RiemannModel supplied by the caller's trainer or load().

    Notes:
        rundown(window) returns (Prediction, original_window). Inherited
        feed()/step()/spit() work with the same registered tubes. Invalid windows
        skip all model computation and produce an explicit invalid Prediction.
    """

    def __init__(self, model: RiemannModel) -> None:
        """Register ordinary transformation functions using the existing architecture."""

        super().__init__()
        self.model = model
        self.adapter = WindowAdapter(
            model.config.expected_channels,
            model.config.sample_count(model.config.window_seconds),
        )

        def validate_window(window: EEGWindow) -> tuple[None, InferenceFrame]:
            """Validate provenance and prepare one samples-by-channels batch."""

            model.validate_window(window)
            frame = InferenceFrame(window)
            if window.valid:
                frame.data = self.adapter.transform(window)
            return None, frame

        def estimate_covariance(frame: InferenceFrame) -> tuple[None, InferenceFrame]:
            """Convert valid windows to regularized channel covariance matrices."""

            if frame.data is not None:
                frame.covariances = model.covariance.transform(frame.data)
            return None, frame

        def map_tangent(frame: InferenceFrame) -> tuple[None, InferenceFrame]:
            """Use the fixed training reference; never refit on live data."""

            if frame.covariances is not None:
                frame.features = model.tangent.transform(frame.covariances)
            return None, frame

        def predict(frame: InferenceFrame) -> tuple[Prediction, EEGWindow]:
            """Return probabilities with timing, or an explicitly rejected result."""

            probabilities = None
            if frame.features is not None:
                probabilities = model.classifier.predict_proba(frame.features)[0]
            result = Prediction(
                frame.window, model.model_id, model.classes, probabilities
            )
            return result, frame.window

        self.add_tube(validate_window)
        self.add_tube(estimate_covariance)
        self.add_tube(map_tangent)
        self.add_tube(predict)
