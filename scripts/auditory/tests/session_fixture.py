"""Self-contained fixtures for the session and producer tests.

Not collected as a test module (the file name does not start with ``test_``): it
exists so the session and producer tests build their own premise instead of
inheriting the environment (plan section 6.4 item 1). Everything - the two
candidate WAVs, their offline envelopes, the decoder - is generated inside the
caller's temporary directory, so the tests run with no dataset and no models
directory present.

The envelopes are written by the same offline code the demo uses
(``scripts/auditory/envelopes.py``) and then verified at load, source hash
included, exactly as a session does at start-up.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile

from nova2026.auditory.config import AuditoryConfig
from nova2026.auditory.data import AuditoryTrial
from scripts.auditory.envelopes import convert_audio, save_envelope
from scripts.auditory.synthetic import synthetic_trial
from scripts.auditory.train import train


def build_model(seconds: float = 20.0):
    """Fit a decoder on two fixtures the replay trial does not share."""

    model, _ = train(
        [synthetic_trial("fixture-training", seconds=int(seconds))],
        [synthetic_trial("fixture-validation", seconds=int(seconds))],
        alphas=(100.0,),
    )
    return model


def build_fixture(
    directory: str | Path, *, seconds: float = 20.0, labels: int = 0
):
    """Write a replayable fixture and its two offline envelopes.

    Args:
        directory: Temporary directory to write into; created if absent.
        seconds: Fixture length in seconds.
        labels: Label every sample carries. The session never reads it; the
            label-isolation test changes it and expects identical decisions.

    Returns:
        ``(model, trial, envelope_paths)``.
    """

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = AuditoryConfig()
    trial = synthetic_trial("session", seconds=int(seconds))
    names = []
    for index, identity in enumerate(("a", "b")):
        target = directory / f"candidate_{identity}.wav"
        samples = np.clip(trial.audio[:, index], -1.0, 1.0)
        wavfile.write(
            target, round(trial.audio_rate), (samples * 32767).astype(np.int16)
        )
        names.append(target.name)
        envelope, timestamps, metadata = convert_audio(target, config)
        save_envelope(
            directory / f"candidate_{identity}.npz", envelope, timestamps, metadata
        )
    trial.group = "|".join(names)
    if labels:
        trial.labels = np.full(len(trial.eeg), int(labels), dtype=int)
    paths = (directory / "candidate_a.npz", directory / "candidate_b.npz")
    return build_model(seconds), trial, paths


def relabelled(trial: AuditoryTrial, label: int) -> AuditoryTrial:
    """The same trial with every label replaced, for the label-isolation test."""

    return AuditoryTrial(
        trial.eeg,
        trial.timestamps,
        trial.audio,
        trial.audio_rate,
        np.full(len(trial.eeg), int(label), dtype=int),
        trial.subject,
        trial.trial_id,
        trial.channel_names,
        trial.reference,
        trial.upstream_processing,
        trial.group,
    )
