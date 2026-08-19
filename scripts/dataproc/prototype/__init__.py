"""Prototype implementation of trial-level EEG preprocessing and engagement features."""

from .pipelines import PIPELINES, apply_pipeline, attentivu_pipeline, default_pipeline
from .spectral import compute_trial_spectral_features

__all__ = [
    "PIPELINES",
    "apply_pipeline",
    "attentivu_pipeline",
    "compute_trial_spectral_features",
    "default_pipeline",
]
