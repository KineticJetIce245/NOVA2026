"""Portable fitted components and their complete input contract."""

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

from scripts.dataproc.streaming.config import StreamConfig
from scripts.dataproc.streaming.window import EEGWindow

from .covariance import CovarianceEstimator
from .tangent import TangentSpace


class RiemannModel:
    """Bundle components fitted by an external trainer; never train implicitly.

    Args:
        config: Exact live configuration used to generate training windows.
        tangent: TangentSpace fitted on training covariances only.
        classifier: Fitted sklearn LogisticRegression on those tangent features.
        label_definition: Plain description of what each class means.
        artifact_id: Required spatial operator ID, or None for uncorrected EEG.
        ridge: Same covariance ridge used when generating training features.
        training_info: Optional split/seed/data provenance supplied by the trainer.
    """

    def __init__(
        self,
        config: StreamConfig,
        tangent: TangentSpace,
        classifier: LogisticRegression,
        label_definition: str,
        artifact_id: str | None = None,
        ridge: float = 1e-6,
        training_info: dict | None = None,
    ) -> None:
        """Validate fitted dimensions and retain the preprocessing contract."""

        check_is_fitted(
            classifier, ["coef_", "intercept_", "classes_", "n_features_in_"]
        )
        channels = len(config.expected_channels)
        features = channels * (channels + 1) // 2
        if tangent.reference is None or tangent.reference.shape != (channels, channels):
            raise ValueError(
                "A fitted tangent reference must match the EEG channel count."
            )
        if (
            vars(classifier)["n_features_in_"] != features
            or classifier.coef_.shape[1] != features
        ):
            raise ValueError("Classifier dimensions do not match tangent features.")
        classes = np.asarray(classifier.classes_.tolist())
        if classes.dtype.kind not in "biufUS" or len(classes) < 2:
            raise ValueError("Use at least two numeric or string class labels.")
        expected_rows = 1 if len(classes) == 2 else len(classes)
        if classifier.coef_.shape[0] != expected_rows or np.asarray(
            classifier.intercept_
        ).shape != (expected_rows,):
            raise ValueError("Classifier coefficients do not match its classes.")
        if not np.all(np.isfinite(classifier.coef_)) or not np.all(
            np.isfinite(classifier.intercept_)
        ):
            raise ValueError("Classifier weights must be finite.")
        if not label_definition.strip():
            raise ValueError("Describe the prediction labels explicitly.")

        self.config = config.updated()
        self.contract = config.window_contract()
        self.tangent = tangent
        self.classifier = classifier
        self.covariance = CovarianceEstimator(ridge)
        self.label_definition = label_definition
        self.artifact_id = artifact_id
        self.training_info = dict(training_info or {})
        self.classes = classes
        self.model_id = self._identifier()

    def _identifier(self) -> str:
        """Identify learned weights, class order and the preprocessing contract."""

        assert self.tangent.reference is not None
        metadata = json.dumps(
            {
                "contract": self.contract,
                "artifact_id": self.artifact_id,
                "classes": self.classes.tolist(),
                "labels": self.label_definition,
                "ridge": self.covariance.ridge,
                "reference_method": "log-euclidean",
            },
            sort_keys=True,
            allow_nan=False,
        ).encode()
        arrays = [
            self.tangent.reference,
            self.classifier.coef_,
            self.classifier.intercept_,
        ]
        content = b"".join(
            np.asarray(array, dtype=np.float64).tobytes() for array in arrays
        )
        return hashlib.sha256(metadata + content).hexdigest()

    def validate_window(self, window: EEGWindow) -> None:
        """Reject a different preprocessing contract, channel order or correction."""

        if self._identifier() != self.model_id:
            raise ValueError("Fitted components changed after model construction.")
        if not np.array_equal(self.classifier.classes_, self.classes):
            raise ValueError("Classifier class order changed after model construction.")
        samples = self.config.sample_count(self.config.window_seconds)
        if window.data.shape != (samples, len(self.config.expected_channels)):
            raise ValueError("Window dimensions differ from model training.")
        if window.timestamps.shape != (samples,) or not np.all(
            np.isfinite(window.timestamps)
        ):
            raise ValueError("Prediction windows need finite sample timestamps.")
        if np.any(np.diff(window.timestamps) <= 0):
            raise ValueError("Prediction timestamps must increase.")

        if (
            window.contract != self.contract
            or window.channel_names != self.config.expected_channels
        ):
            raise ValueError(
                "Window preprocessing/channel contract differs from the model."
            )
        if window.artifact_id != self.artifact_id:
            raise ValueError("Window spatial correction differs from model training.")

    def save(self, path: Path) -> None:
        """Save numeric fitted state and JSON metadata without pickle or overwrite."""

        if self._identifier() != self.model_id:
            raise ValueError("Fitted components changed; construct a new model bundle.")
        metadata = {
            "schema": 1,
            "config": self.config.to_dict(),
            "label_definition": self.label_definition,
            "artifact_id": self.artifact_id,
            "ridge": self.covariance.ridge,
            "training_info": self.training_info,
            "model_id": self.model_id,
            "reference_method": "log-euclidean",
        }
        assert self.tangent.reference is not None
        with Path(path).open("xb") as stream:
            np.savez_compressed(
                stream,
                reference=self.tangent.reference,
                coef=self.classifier.coef_,
                intercept=self.classifier.intercept_,
                classes=self.classes,
                metadata=np.array(json.dumps(metadata, allow_nan=False)),
            )

    @classmethod
    def load(cls, path: Path) -> "RiemannModel":
        """Restore fitted numeric state; no fitting or executable pickle loading occurs."""

        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"].item()))
            if (
                metadata["schema"] != 1
                or metadata["reference_method"] != "log-euclidean"
            ):
                raise ValueError(
                    "Unsupported model format or tangent reference method."
                )
            tangent = TangentSpace()
            tangent.set_reference(saved["reference"])
            classifier = LogisticRegression()
            classifier.classes_ = saved["classes"].copy()
            classifier.coef_ = saved["coef"].copy()
            classifier.intercept_ = saved["intercept"].copy()
            vars(classifier)["n_features_in_"] = classifier.coef_.shape[1]
            classifier.n_iter_ = np.array([0], dtype=np.int32)
            model = cls(
                StreamConfig.from_dict(metadata["config"]),
                tangent,
                classifier,
                metadata["label_definition"],
                metadata["artifact_id"],
                metadata["ridge"],
                metadata["training_info"],
            )
        if model.model_id != metadata["model_id"]:
            raise ValueError("Saved model identifier does not match its contents.")
        return model
