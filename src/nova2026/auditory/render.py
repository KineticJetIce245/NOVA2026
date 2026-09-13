"""Render the two candidates into the stereo file the browser actually plays.

Plan section 3.3 fixes the long-standing presentation - **left channel = candidate

A, right channel = candidate B** - and section 3.15 records that the *target*

presentation is one mixture in both ears, with a differently-weighted per-channel

mix as the interim route that needs no change to the vendored frontend. The two

routes differ in sample arithmetic only, so they live here as named modes rather

than as two copies of a mixer:

``dichotic``

    ``L = A`` and ``R = B``. This is what the KU Leuven data is (the recorded

    trials are ear-separated), what the step-5 decoder was fitted and evaluated

    on, and what the user's recorded ANT sessions are (plan section 3.12). It is

    the mode wired for the KU Leuven replay.

``crossmix``

    ``L = (A + w*B)/(1 + w)`` and ``R = (B + w*A)/(1 + w)`` with ``w < 1``: each

    ear hears both voices and A leads in the left. This is the interim form of

    section 3.15's target - it needs no frontend change, because the existing

    ``ChannelSplitter`` + two ``GainNode`` path still attenuates one channel per

    ear - but it is *not* the target either: the frontend then attenuates a mix,

    not a source, so the dB numbers no longer mean "candidate A's level".

The rendered file is derived output: it is written under ``output/`` (git-ignored)

and carries no measured quantity, only audio the caller already has. The

per-channel SHA256 values in the report are there so a report can say *which*

samples were served without shipping them.

"""

from __future__ import annotations

import hashlib

from pathlib import Path

from typing import Any

import numpy as np

from scipy.io import wavfile

INT16_PEAK = 32767

"""Full scale of the rendered PCM; the same convention as the step-8 fixture."""

PRESENTATION_MODES = ("dichotic", "crossmix")

"""Presentation routes this module can render. Plan sections 3.3, 3.15 and D-21."""

DEFAULT_PRESENTATION = "dichotic"

"""The mode the KU Leuven replay is wired with, and the reason is in the docstring."""

DEFAULT_CROSSMIX_WEIGHT = 0.25

"""Level of the *other* candidate in each channel, relative to the leading one."""

def presentation_mode(name: str = DEFAULT_PRESENTATION) -> str:
    """Validate one mode name and return it.

    Raises:

        ValueError: If ``name`` is not one of :data:`PRESENTATION_MODES`. An

            unknown mode is refused instead of silently rendering something the

            report does not name.

    """

    if name not in PRESENTATION_MODES:
        raise ValueError(

            f"presentation must be one of {PRESENTATION_MODES}, not {name!r}."

        )

    return name

def mix_candidates(

    candidates: np.ndarray,

    *,

    presentation: str = DEFAULT_PRESENTATION,

    crossmix_weight: float = DEFAULT_CROSSMIX_WEIGHT,

) -> np.ndarray:

    """Build the two output channels from the two candidate waveforms.

    Args:

        candidates: ``(samples, 2)`` array; column 0 is candidate A, column 1 is

            candidate B. Values are interpreted in ``[-1, 1]``, the normalized

            convention ``AuditoryTrial`` enforces for both candidates.

        presentation: ``dichotic`` or ``crossmix`` (see the module docstring).

        crossmix_weight: Weight of the non-leading candidate in ``crossmix``. Must

            be finite and in ``[0, 1]``: 0 degenerates to ``dichotic`` and 1 makes

            both ears identical, which would be a different decision than the one

            this parameter is for.

    Returns:

        ``(samples, 2)`` float array of the left and right channels, clipped to

        ``[-1, 1]``.

    Raises:

        ValueError: If the array is not finite and two-channel, if it is empty, or

            if the mode or the weight is out of range.

    """

    mode = presentation_mode(presentation)

    if (

        isinstance(crossmix_weight, bool)

        or not isinstance(crossmix_weight, (int, float))

        or not np.isfinite(crossmix_weight)

        or not 0.0 <= float(crossmix_weight) <= 1.0

    ):
        raise ValueError("crossmix_weight must be a finite number in [0, 1].")

    samples = np.asarray(candidates, dtype=float)

    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError(

            "candidates must be a (samples, 2) array of candidate A and candidate B."

        )

    if samples.shape[0] == 0:
        raise ValueError("candidates must not be empty.")

    if not np.all(np.isfinite(samples)):
        raise ValueError("candidates must be finite; a non-finite sample is not audio.")

    if mode == "dichotic":
        return np.clip(samples, -1.0, 1.0)

    weight = float(crossmix_weight)

    leading, trailing = samples[:, 0], samples[:, 1]

    scale = 1.0 / (1.0 + weight)

    left = (leading + weight * trailing) * scale

    right = (trailing + weight * leading) * scale

    return np.clip(np.stack((left, right), axis=1), -1.0, 1.0)

def to_int16(channels: np.ndarray) -> np.ndarray:
    """Scale channel values in ``[-1, 1]`` to int16 PCM at the file's own level.

    The mapping is absolute - ``1.0`` is full scale - rather than relative to the

    array's peak, because the two candidates' relative levels are part of the

    stimulus: a peak-normalizing conversion would silently undo the balance

    between them, and in ``crossmix`` it would undo the mix itself. Converted

    trials are already normalized to ``[-1, 1]`` (``AuditoryTrial`` refuses

    anything else), so this is the whole conversion.

    """

    values = np.asarray(channels, dtype=float)

    scaled = values * INT16_PEAK

    return np.clip(scaled, -INT16_PEAK - 1, INT16_PEAK).astype(np.int16)

def channel_sha256(channels: np.ndarray) -> dict[str, str]:
    """SHA256 of each output channel's PCM, so a report can name what was served."""

    return {

        name: hashlib.sha256(np.ascontiguousarray(channels[:, index]).tobytes()).hexdigest()

        for index, name in enumerate(("left", "right"))

    }

def render_stereo(

    candidates: np.ndarray,

    path: str | Path,

    *,

    sample_rate: int,

    presentation: str = DEFAULT_PRESENTATION,

    crossmix_weight: float = DEFAULT_CROSSMIX_WEIGHT,

) -> dict[str, Any]:

    """Render the two candidates to one stereo WAV and report what was written.

    Args:

        candidates: ``(samples, 2)`` array; column 0 is candidate A, column 1 is

            candidate B.

        path: Destination file; its parent directory is created if needed. The

            file is 16-bit PCM, the format the dataset's own stimuli use.

        sample_rate: Output sample rate in Hz, taken from the recording rather

            than assumed (plan section 1 fact 11: 44.1 kHz dataset audio, 48 kHz

            live audio).

        presentation: Presentation route, as in :func:`mix_candidates`.

        crossmix_weight: Weight of the non-leading candidate in ``crossmix``.

    Returns:

        A JSON-safe report: the file name, the mode and weight, the sample count

        and rate, per-channel peak in int16 units, and per-channel SHA256.

    Raises:

        ValueError: If ``sample_rate`` is not a positive integer, or for any

            reason :func:`mix_candidates` raises.

    """

    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)):
        raise ValueError("sample_rate must be a positive integer number of hertz.")

    if int(sample_rate) <= 0:
        raise ValueError("sample_rate must be a positive integer number of hertz.")

    mode = presentation_mode(presentation)

    channels = mix_candidates(

        candidates, presentation=mode, crossmix_weight=crossmix_weight

    )

    pcm = to_int16(channels)

    target = Path(path)

    target.parent.mkdir(parents=True, exist_ok=True)

    wavfile.write(target, int(sample_rate), pcm)

    return {

        "path": str(target),

        "file": target.name,

        "presentation": mode,

        "crossmix_weight": float(crossmix_weight) if mode == "crossmix" else None,

        "sample_rate": int(sample_rate),

        "samples": int(pcm.shape[0]),

        "seconds": float(pcm.shape[0]) / float(sample_rate),

        "channels": list(("A", "B")) if mode == "dichotic" else ["A+B", "B+A"],

        "peak_int16": {

            name: int(np.max(np.abs(pcm[:, index]))) if pcm.size else 0

            for index, name in enumerate(("left", "right"))

        },

        "bytes": int(target.stat().st_size),

        "sha256": channel_sha256(pcm),

    }

__all__ = [

    "DEFAULT_CROSSMIX_WEIGHT",

    "DEFAULT_PRESENTATION",

    "INT16_PEAK",

    "PRESENTATION_MODES",

    "channel_sha256",

    "mix_candidates",

    "presentation_mode",

    "render_stereo",

    "to_int16",

]
