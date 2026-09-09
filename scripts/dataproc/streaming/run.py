"""Run identity and explicit selection of calibration, baseline, and model files."""

import re
from pathlib import Path


class RunSpec:
    """Describe one uniquely named recording and its selected inputs.

    Args:
        subject: Subject identifier; avoid personally identifying information.
        session: Session identifier shared by related recordings.
        run_id: Unique run name within the subject and session.
        role: artifact_calibration, baseline, labeled_training, or trial.
        artifact_path: Saved spatial operator to apply, or None for raw processing.
        baseline_path: Explicitly selected baseline file, e.g. its run.sqlite.
        model_path: Explicitly selected model, if needed downstream.

    Notes:
        Selecting a model or baseline records provenance; it does not load a
        classifier or perform baseline normalization. Those remain downstream.
    """

    def __init__(
        self,
        subject: str,
        session: str,
        run_id: str,
        role: str,
        artifact_path: Path | None = None,
        baseline_path: Path | None = None,
        model_path: Path | None = None,
    ) -> None:
        """Validate identifiers before constructing any output paths."""

        for value in (subject, session, run_id):
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value):
                raise ValueError(
                    "Run identifiers must use letters, digits, underscores, or hyphens."
                )

        if role not in (
            "artifact_calibration",
            "baseline",
            "labeled_training",
            "trial",
        ):
            raise ValueError("Unsupported recording role.")

        if role == "artifact_calibration" and artifact_path is not None:
            raise ValueError(
                "Calibration must be recorded without an existing correction."
            )

        self.subject = subject
        self.session = session
        self.run_id = run_id
        self.role = role
        self.artifact_path = artifact_path
        self.baseline_path = baseline_path
        self.model_path = model_path

    def to_dict(self) -> dict:
        """Return JSON-compatible run identity and selected source paths."""

        return {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(self).items()
        }
