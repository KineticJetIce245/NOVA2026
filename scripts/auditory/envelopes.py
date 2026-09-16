"""Precompute the reference speech envelopes a session refuses to run without.

The demo decodes attention against an envelope of the audio that was actually
played. Computing it inside the acquisition thread is forbidden (plan §3.2), and
computing it lazily at session start would silently hide an audio/integrity
problem, so every stimulus gets an envelope here first:

    python -B -m scripts.auditory.envelopes \
        --audio datasets/AAD-KULeuven/stimuli/part1_track1_dry.wav [more...] \
        --out datasets/audio

One ``<audio_stem>.npz`` per input, holding ``envelope`` (float32, N x 1 at
``sample_rate`` Hz), ``timestamps`` (float64 seconds from audio sample 0) and
``metadata`` (a JSON string: source path, source SHA256, source and feature
rates, band, method, generation time, generator version). Compressed ``.npz``
and JSON only - never pickle.

Envelopes are derived from a dataset, so they live under the git-ignored
``datasets/audio/``. The recipe is the committed artifact; the megabytes are not.

Inputs must be mono, or two-channel, which ``read_audio`` downmixes to mono the
same way the decoder path does. More than two channels are refused rather than
silently averaged, because a 5.1 file is almost certainly not what was played.
"""

import argparse
import json
import os
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from nova2026.auditory.audio import read_audio
from nova2026.auditory.config import AuditoryConfig
from nova2026.auditory.envelopes import (
    GENERATOR_VERSION,
    EnvelopeVerificationError,
    envelope_timestamps,
    extract_envelope,
    load_envelope,
    resolve_source_path,
    source_sha256,
)
from scripts.auditory.outputs import guard_outputs

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "datasets" / "audio"


def channel_count(path):
    """Channels in a WAV header, so a multichannel file is refused before it is mixed.

    ``read_audio`` averages every channel, which is right for the stereo playlist
    and wrong for anything wider: a 5.1 file would silently become a different
    signal than the one that was presented. The header is read, not the samples.
    """

    with wave.open(str(path), "rb") as stream:
        return int(stream.getnchannels())


def recorded_source_path(path, repo=None):
    """How ``source_path`` is stored: repository-relative when the source is inside.

    An envelope outlives the machine that wrote it, and an absolute path does not
    survive a different drive letter, a different mount point or a different
    operating system. A source inside the repository is therefore recorded
    relative to the repository root -- ``datasets/AAD-KULeuven/stimuli/x.wav`` --
    which :func:`~nova2026.auditory.envelopes.load_envelope` re-roots at whichever
    copy is reading it (via the working directory first, then the envelope's own
    ancestors).

    A source from outside the repository has no such anchor and stays absolute.
    The spelling is POSIX on every platform (``as_posix``), so an envelope written
    on Windows verifies on macOS and the other way round.
    """

    anchor = Path(REPO if repo is None else repo).resolve()
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(anchor).as_posix()
    except ValueError:
        return str(resolved)


def metadata_for(path, samples, rate, config, generated_at):
    """The provenance record stored beside the envelope.

    ``sample_rate``, ``band`` and ``envelope_method`` come from the config's own
    ``to_dict()``, so what is recorded cannot drift from what a session uses.
    """

    recorded = config.to_dict()
    return {
        "source_path": recorded_source_path(path),
        "source_sha256": source_sha256(path),
        "source_rate": int(rate),
        "source_samples": int(len(samples)),
        "audio_rate": int(rate),
        "sample_rate": recorded["sample_rate"],
        "band": [float(value) for value in recorded["band"]],
        "envelope_method": recorded["envelope_method"],
        "generated_at": generated_at,
        "generator_version": GENERATOR_VERSION,
    }


def convert_audio(path, config, *, generated_at=None):
    """Read one audio file and return its ``(envelope, timestamps, metadata)``.

    Raises:
        ValueError: If the file is missing, has more than two channels, or does
            not carry samples.
    """

    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Audio file is missing: {path}")
    channels = channel_count(path)
    if channels > 2:
        raise ValueError(
            f"{path} has {channels} channels; supply mono or stereo audio "
            "(a wider file is not what a two-candidate session played)."
        )
    samples, rate = read_audio(path)
    if not len(samples):
        raise ValueError(f"Audio file holds no samples: {path}")
    envelope = extract_envelope(samples, rate, config)
    timestamps = envelope_timestamps(len(envelope), config.sample_rate)
    stamp = generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    return envelope, timestamps, metadata_for(path, samples, rate, config, stamp)


def save_envelope(path, envelope, timestamps, metadata):
    """Write one envelope atomically: a half-written npz must never be loadable."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    try:
        with open(temporary, "wb") as stream:
            np.savez_compressed(
                stream,
                envelope=np.asarray(envelope, dtype=np.float32),
                timestamps=np.asarray(timestamps, dtype=np.float64),
                metadata=json.dumps(metadata, sort_keys=True),
            )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path.stat().st_size


def expand_inputs(files, directories, patterns):
    """Turn the CLI's file/directory arguments into a sorted list of inputs.

    One pattern is broadcast to every directory; giving several requires exactly
    one per directory, so a mismatch is reported instead of silently dropping a
    search root.
    """

    selected = [Path(item) for item in files]
    if len(patterns) == 1:
        patterns = patterns * len(directories)
    elif len(patterns) != len(directories):
        raise ValueError(
            f"{len(patterns)} --pattern values for {len(directories)} --audio-dir "
            "directories; supply one pattern or one per directory."
        )
    for directory, pattern in zip(directories, patterns):
        root = Path(directory)
        if not root.is_dir():
            raise ValueError(f"Audio directory is missing: {root}")
        selected.extend(sorted(root.glob(pattern)))
    unique = {}
    for item in selected:
        unique.setdefault(item.resolve(), item)
    return [unique[key] for key in sorted(unique)]


def check_stems(paths):
    """Refuse two inputs that would write the same ``<stem>.npz``."""

    stems = {}
    for path in paths:
        stems.setdefault(path.stem, []).append(str(path))
    clashes = {stem: names for stem, names in stems.items() if len(names) > 1}
    if clashes:
        raise ValueError(
            "Two inputs share a file stem and would collide: "
            + "; ".join(f"{stem} <- {', '.join(names)}" for stem, names in clashes.items())
        )


def verify_outputs(paths, config):
    """Re-verify envelopes already on disk, hashes included, and report failures."""

    failed = []
    for path in paths:
        try:
            recorded = recorded_path(path)
            # Resolve the way a session would rather than trusting the recorded
            # path to be relative to the caller's working directory, so --verify
            # says the same thing whether it is run from the root or elsewhere.
            resolved = resolve_source_path(recorded, path)
            record = load_envelope(
                path, config, audio_path=resolved if resolved else recorded
            )
        except EnvelopeVerificationError as error:
            failed.append(f"{path}: {error}")
            continue
        print(f"ok    {path}  {len(record)} samples @ {record.sample_rate:g} Hz")
    for line in failed:
        print(f"FAIL  {line}", file=sys.stderr)
    return 1 if failed else 0


def recorded_path(envelope_path):
    """The source path an existing envelope records, for ``--verify`` hashing."""

    with np.load(envelope_path, allow_pickle=False) as stored:
        metadata = json.loads(str(stored["metadata"]))
    return Path(metadata["source_path"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", nargs="+", default=[], help="audio files to convert")
    parser.add_argument(
        "--audio-dir",
        nargs="+",
        default=[],
        help="directories whose WAV files are converted",
    )
    parser.add_argument(
        "--pattern",
        nargs="+",
        default=["*.wav"],
        help="glob applied to each --audio-dir (one value, or one per directory)",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output directory")
    parser.add_argument(
        "--force", action="store_true", help="overwrite envelopes that already exist"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="only re-check the envelopes in --out against their source audio",
    )
    args = parser.parse_args(argv)

    config = AuditoryConfig()
    output = Path(args.out)
    if args.verify:
        if not output.is_dir():
            parser.error(f"Nothing to verify: {output} does not exist.")
        return verify_outputs(sorted(output.glob("*.npz")), config)

    try:
        inputs = expand_inputs(args.audio, args.audio_dir, args.pattern)
    except ValueError as error:
        parser.error(str(error))
    if not inputs:
        parser.error("Supply --audio files, --audio-dir directories, or --verify.")
    try:
        check_stems(inputs)
    except ValueError as error:
        parser.error(str(error))

    targets = [output / f"{item.stem}.npz" for item in inputs]
    try:
        guard_outputs(targets, force=args.force)
    except FileExistsError as error:
        parser.error(str(error))

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    total = 0
    for source, target in zip(inputs, targets):
        try:
            envelope, timestamps, metadata = convert_audio(
                source, config, generated_at=stamp
            )
        except ValueError as error:
            parser.error(str(error))
        size = save_envelope(target, envelope, timestamps, metadata)
        total += size
        span = len(envelope) / config.sample_rate
        print(
            f"wrote {target}  shape=({len(envelope)}, 1)  dtype={envelope.dtype}  "
            f"{span:.1f} s @ {config.sample_rate:g} Hz  {size / 1024:.1f} KiB"
        )
    print(
        f"{len(targets)} envelope(s), {total} bytes ({total / 1024:.1f} KiB) "
        f"in {output}  band={config.to_dict()['band']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
