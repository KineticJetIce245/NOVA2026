"""Continuous speech-envelope extraction with bounded retained state.

Two entry points share one implementation:

* :class:`EnvelopeExtractor` is the runtime path. It is causal and chunk-wise:
  state carries across ``feed`` calls, so a live stream and a whole recording
  produce the same grid.
* :func:`extract_envelope` is the offline path used to precompute the reference
  envelopes under ``datasets/audio/``. It is deliberately a thin wrapper over
  the very same extractor - the offline numbers must be the runtime numbers, and
  a second implementation would be free to drift from them.

A session refuses to start on a missing or mismatched envelope, so the recorded
provenance is checked against both the audio on disk and the live
:class:`~nova2026.auditory.config.AuditoryConfig` by :func:`load_envelope`.
"""

import hashlib
import json
import math
from pathlib import Path
from typing import cast

import numpy as np
from scipy.signal import butter, sosfilt

# Bumped whenever the extractor's numbers can change. An envelope recording an
# older version is refused rather than silently decoded with a different feature.
GENERATOR_VERSION = "1"

# 1 MiB. Big enough that hashing a 15-minute stimulus is one syscall per megabyte.
_HASH_CHUNK = 1 << 20

# The runtime grid, restated for verification: index i is i / sample_rate seconds
# from audio sample 0.
_TIMESTAMP_ORIGIN = 0.0


class EnvelopeExtractor:
    """Rectify speech, smooth and band-limit it, then resample continuously."""

    def __init__(self, audio_rate, config):
        self.audio_rate = audio_rate
        self.config = config
        lowpass = butter(8, 20, fs=audio_rate, output="sos")
        bandpass = butter(3, config.band, btype="bandpass", fs=audio_rate, output="sos")
        self.sos = np.vstack([cast(np.ndarray, lowpass), cast(np.ndarray, bandpass)])
        self.reset()

    def reset(self):
        self.state = np.zeros((len(self.sos), 2, 2))
        self.input_count = 0
        self.output_count = 0
        self.previous = None

    def feed(self, samples):
        samples = np.asarray(samples, dtype=float)
        if samples.ndim != 2 or samples.shape[1] != 2:
            raise ValueError("Expected two audio candidate columns.")
        if not np.all(np.isfinite(samples)):
            raise ValueError("Audio samples must be finite.")
        if len(samples) == 0:
            return np.empty((0, 2))
        filtered, self.state = sosfilt(self.sos, np.abs(samples), axis=0, zi=self.state)
        # Band-limit before sampling the common grid. Keep the preceding sample
        # so interpolation across a chunk boundary uses the same two endpoints.
        positions = self.input_count + np.arange(len(filtered))
        if self.previous is not None:
            positions = np.concatenate([[self.input_count - 1], positions])
            filtered = np.vstack([self.previous, filtered])
        self.input_count += len(samples)
        last_position = self.input_count - 1
        output_end = (
            int(np.floor(last_position * self.config.sample_rate / self.audio_rate)) + 1
        )
        targets = (
            np.arange(self.output_count, output_end)
            * self.audio_rate
            / self.config.sample_rate
        )
        result = np.empty((len(targets), 2))
        for candidate in range(2):
            result[:, candidate] = np.interp(targets, positions, filtered[:, candidate])
        self.previous = filtered[-1].copy()
        self.output_count = output_end
        return result


def envelope_timestamps(count, sample_rate):
    """Seconds from audio sample 0 for the first ``count`` envelope samples.

    The grid is ``i * audio_rate / sample_rate`` in the extractor, i.e. exact
    multiples of ``1 / sample_rate``, so this is a closed form of the same thing
    and not a second convention.
    """

    if count < 0:
        raise ValueError("An envelope cannot have a negative length.")
    if not math.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")
    return _TIMESTAMP_ORIGIN + np.arange(count, dtype=np.float64) / sample_rate


def extract_envelope(samples, audio_rate, config):
    """Whole-file envelope, identical to feeding the same signal in chunks.

    ``samples`` is one mono signal (``(n,)``), or the ``(n, 2)`` candidate array
    the runtime extractor takes.

    Mono handling: the extractor works on two candidate columns because that is
    what a live session decodes. A single stimulus is therefore fed as **two
    identical columns**, and column 0 is kept. The columns never mix - each is
    filtered and interpolated on its own - so column 0 of a doubled mono signal
    is exactly what a hypothetical one-column extractor would return, and the
    runtime component stays the tested one. Recorded here because the choice is
    invisible in the numbers and would be hard to recover from them.

    Args:
        samples: ``(n,)`` mono, or ``(n, 2)`` candidates.
        audio_rate: Sample rate of ``samples`` in Hz.
        config: :class:`~nova2026.auditory.config.AuditoryConfig` supplying the
            feature rate and band.

    Returns:
        ``(m, 1)`` float32 array; ``m == floor((n - 1) * config.sample_rate /
        audio_rate) + 1``, which is the count the runtime extractor reports.
    """

    values = np.asarray(samples, dtype=float)
    if values.ndim == 1:
        values = np.column_stack([values, values])
    elif values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Expected a mono signal or exactly two candidate columns.")
    if not math.isfinite(audio_rate) or audio_rate <= 0:
        raise ValueError("Audio rate must be positive.")
    values = np.ascontiguousarray(values)
    envelope = EnvelopeExtractor(audio_rate, config).feed(values)
    return np.ascontiguousarray(envelope, dtype=np.float32)[:, :1]


def source_sha256(path):
    """SHA256 of a source file, streamed so a 100 MB stimulus costs 100 MB of RAM at most."""

    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(_HASH_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


class EnvelopeVerificationError(RuntimeError):
    """A stored envelope is missing, unreadable, or does not match this session.

    One exception type for every refusal. The session start path catches it and
    stops; it never falls back to computing an envelope at run time.
    """


class EnvelopeRecord:
    """A verified envelope with its timestamps and its recorded provenance."""

    def __init__(self, envelope, timestamps, metadata):
        self.envelope = envelope
        self.timestamps = timestamps
        self.metadata = metadata

    @property
    def source_path(self):
        """The audio file this envelope was derived from, as recorded."""

        return self.metadata["source_path"]

    @property
    def sample_rate(self):
        """Feature rate of ``envelope`` and ``timestamps``, in Hz."""

        return float(self.metadata["sample_rate"])

    def __len__(self):
        return len(self.envelope)


def _expected_parameters(config):
    recorded = config.to_dict()
    return {
        "sample_rate": recorded["sample_rate"],
        "band": [float(value) for value in recorded["band"]],
        "envelope_method": recorded["envelope_method"],
        "generator_version": GENERATOR_VERSION,
    }


def _as_band(value):
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return [float(value[0]), float(value[1])]
        except (TypeError, ValueError):
            return None
    return None


def _recorded_spellings(recorded):
    """The recorded path, plus a POSIX spelling when it was written with backslashes.

    A relative path written on Windows (``datasets\\a\\b.wav``) is a *single file
    name* on POSIX, where a backslash is an ordinary character. Both spellings
    are therefore tried, so an envelope written on one platform still resolves on
    the other. Neither spelling is trusted on its own: what resolves is re-hashed.
    """

    recorded = str(recorded)
    spellings = [recorded]
    if "\\" in recorded:
        normalised = recorded.replace("\\", "/")
        if normalised != recorded:
            spellings.append(normalised)
    return spellings


def source_candidates(recorded, envelope_path):
    """Where an envelope's recorded source may be on *this* machine, in order.

    An envelope outlives the machine that wrote it, so the recorded path is a
    hint and never a fact. The candidates are:

    * **as recorded** -- an absolute path on the generating machine, or a path
      relative to the process's working directory, which is the same thing when
      the record is relative to the repository root and the process runs there;
    * **the envelope's grandparent** -- for the shipped ``datasets/audio/``, the
      directory that holds ``datasets/`` (a record relative to ``datasets/``);
    * **the envelope's great-grandparent** -- for ``datasets/audio/``, the
      repository root, which is what ``scripts.auditory.envelopes`` writes;
    * for a record that is **absolute**, and so anchored to a root this machine
      does not have, every trailing component suffix of it under both roots.
      That is what turns ``C:\\Files\\git\\NOVA2026\\datasets\\a\\b.wav`` or
      ``/Volumes/disk/NOVA2026/datasets/a/b.wav`` back into
      ``datasets/a/b.wav``, and it is the case the old single retry could not
      reach: re-rooting an absolute path returns that same absolute path.

    A wrong pick cannot pass unnoticed -- the caller re-hashes whatever was
    found against the recorded ``source_sha256`` -- and the search is a handful
    of ``stat`` calls, run only when the earlier candidates have already missed.
    """

    envelope = Path(envelope_path).resolve()
    roots = []
    for parent in envelope.parents[1:3]:
        if parent != envelope and parent not in roots:
            roots.append(parent)
    candidates = []
    for spelling in _recorded_spellings(recorded):
        path = Path(spelling)
        forms = [path]
        if path.is_absolute():
            tail = path.parts[1:]
            forms.extend(Path(*tail[start:]) for start in range(len(tail)))
        for form in forms:
            for candidate in (form, *(root / form for root in roots)):
                if candidate not in candidates:
                    candidates.append(candidate)
    return candidates


def resolve_source_path(recorded, envelope_path):
    """The first candidate of :func:`source_candidates` that is a file, else None."""

    for candidate in source_candidates(recorded, envelope_path):
        if candidate.is_file():
            return candidate
    return None


def verify_envelope_metadata(metadata, config, *, audio_path=None):
    """Check a stored envelope against the live config and, optionally, the audio.

    Args:
        metadata: The mapping decoded from an ``.npz``'s ``metadata`` entry.
        config: The session's :class:`AuditoryConfig`. ``sample_rate``, ``band``
            and ``envelope_method`` must equal its own ``to_dict()`` values, so
            the recorded parameters cannot drift from the ones the session uses.
        audio_path: The source audio to re-hash. ``None`` leaves path resolution
            to the caller (:func:`load_envelope` resolves the recorded path);
            passing a path that does not exist is an error, never a skip.

    Returns:
        The verified metadata, unchanged.

    Raises:
        EnvelopeVerificationError: On any missing key, wrong type, parameter
            mismatch, generator-version mismatch, or source-hash mismatch.
    """

    expected = _expected_parameters(config)
    if not isinstance(metadata, dict):
        raise EnvelopeVerificationError("Envelope metadata is not an object.")
    required = (
        "source_path",
        "source_sha256",
        "source_rate",
        "source_samples",
        "audio_rate",
        "sample_rate",
        "band",
        "envelope_method",
        "generated_at",
        "generator_version",
    )
    missing = [key for key in required if key not in metadata]
    if missing:
        raise EnvelopeVerificationError(
            "Envelope metadata is missing " + ", ".join(missing) + "."
        )
    band = _as_band(metadata["band"])
    if band is None:
        raise EnvelopeVerificationError("Envelope band must be two numbers.")
    observed = {
        "sample_rate": metadata["sample_rate"],
        "band": band,
        "envelope_method": metadata["envelope_method"],
        "generator_version": metadata["generator_version"],
    }
    for key, wanted in expected.items():
        if observed[key] != wanted:
            raise EnvelopeVerificationError(
                f"Envelope {key} is {observed[key]!r} but this session needs {wanted!r}."
            )
    for key in ("audio_rate", "source_rate", "source_samples"):
        if not isinstance(metadata[key], (int, float)) or isinstance(metadata[key], bool):
            raise EnvelopeVerificationError(f"Envelope {key} must be a number.")
    if not isinstance(metadata["source_path"], str) or not metadata["source_path"]:
        raise EnvelopeVerificationError("Envelope source_path must be a nonempty string.")
    if (
        not isinstance(metadata["source_sha256"], str)
        or len(metadata["source_sha256"]) != 64
    ):
        raise EnvelopeVerificationError("Envelope source_sha256 must be a hex digest.")
    if audio_path is not None:
        path = Path(audio_path)
        if not path.is_file():
            raise EnvelopeVerificationError(f"Source audio is missing: {path}.")
        actual = source_sha256(path)
        if actual != metadata["source_sha256"]:
            raise EnvelopeVerificationError(
                f"Source audio {path} has SHA256 {actual}, but the envelope records "
                f"{metadata['source_sha256']}."
            )
    return metadata


def load_envelope(path, config, *, audio_path=None):
    """Read and verify one precomputed envelope. Never computes a missing one.

    Args:
        path: The ``.npz`` written by ``scripts/auditory/envelopes.py``.
        config: The session's :class:`AuditoryConfig`.
        audio_path: Optional source audio. When it is omitted the file named by
            the envelope's own ``source_path`` is re-hashed: first as recorded,
            then relative to the envelope's grandparent directory (the repository
            root for ``datasets/audio/``, which is what makes a cloned or copied
            tree verifiable). A recorded path that resolves to nothing skips the
            hash check; pass ``audio_path`` to force it.

    Returns:
        :class:`EnvelopeRecord` with ``(m, 1)`` float32 ``envelope``, float64
        ``timestamps`` in seconds from audio sample 0, and ``metadata``.

    Raises:
        EnvelopeVerificationError: If the file is absent, unreadable, lacks the
            ``envelope``/``timestamps``/``metadata`` entries, carries the wrong
            length for its timestamps, or fails verification.
    """

    envelope_path = Path(path)
    if not envelope_path.is_file():
        raise EnvelopeVerificationError(
            f"No precomputed envelope at {envelope_path}; generate it with "
            "`python -B -m scripts.auditory.envelopes --audio <file>`."
        )
    try:
        with np.load(envelope_path, allow_pickle=False) as stored:
            names = set(stored.files)
            absent = {"envelope", "timestamps", "metadata"} - names
            if absent:
                raise EnvelopeVerificationError(
                    f"{envelope_path} is missing " + ", ".join(sorted(absent)) + "."
                )
            envelope = np.asarray(stored["envelope"], dtype=np.float32)
            timestamps = np.asarray(stored["timestamps"], dtype=np.float64)
            raw_metadata = stored["metadata"]
    except EnvelopeVerificationError:
        raise
    except Exception as error:  # unreadable file, corrupt zip, non-JSON metadata
        raise EnvelopeVerificationError(f"Cannot read {envelope_path}: {error}") from error

    try:
        metadata = json.loads(str(raw_metadata))
    except (TypeError, ValueError) as error:
        raise EnvelopeVerificationError(
            f"{envelope_path} does not carry JSON metadata: {error}"
        ) from error
    # The hash is what catches a stimulus that changed under a stale envelope, so
    # it is checked whenever the recording names a file that is here. Envelopes
    # also live in a repository that gets cloned, copied and moved between
    # operating systems, where the absolute path of the generating machine is
    # gone; :func:`source_candidates` re-roots the record at this tree and, for
    # an absolute record, at the repository-relative tail of it. A path that
    # resolves nowhere is reported to the caller as "unchecked", and the CLI's
    # ``--verify`` turns that into a failure rather than a run. What is never
    # allowed is decoding an envelope nothing checked.
    reference = audio_path
    if reference is None and isinstance(metadata, dict):
        recorded = metadata.get("source_path")
        if isinstance(recorded, str):
            reference = resolve_source_path(recorded, envelope_path)
    verify_envelope_metadata(metadata, config, audio_path=reference)
    if envelope.ndim != 2 or envelope.shape[1] != 1:
        raise EnvelopeVerificationError(
            f"Envelope must be (samples, 1); {envelope_path} holds {envelope.shape}."
        )
    if len(envelope) != len(timestamps):
        raise EnvelopeVerificationError(
            f"Envelope and timestamps disagree: {len(envelope)} vs {len(timestamps)}."
        )
    if not np.all(np.isfinite(envelope)):
        raise EnvelopeVerificationError("Envelope contains non-finite samples.")
    expected_rate = metadata["sample_rate"]
    if len(timestamps) > 1 and expected_rate > 0:
        step = float(timestamps[1] - timestamps[0])
        if not math.isclose(step, 1.0 / expected_rate, rel_tol=1e-9, abs_tol=1e-12):
            raise EnvelopeVerificationError(
                f"Timestamps step by {step} s, not 1/{expected_rate} s."
            )
    return EnvelopeRecord(envelope, timestamps, metadata)