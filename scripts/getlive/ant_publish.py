"""Publish one imported ANT session as a loopback LSL outlet, at its real rate.

This is the missing half of the live route: ``scripts/auditory/antneuro.py``
turned the operator's own ``.cnt`` recordings into trials, and this plays one of
them back over LSL so the *live* path (``nova2026.streaming`` pre-flight plus
``Acquire``, and :class:`~nova2026.auditory.session.AttentionSession` through
:class:`~scripts.getlive.ant_source.AntStreamSource`) can be driven with real
recorded EEG. It is a source, not a decoder: it carries the recording's own
microvolt samples, its own channel names and its own clock.

Why the samples are microvolts on the wire. The trial file holds microvolts
already (``scripts/auditory/antneuro.py`` converts the ``.cnt``'s volts by
``x1e+06`` and records ``chain_source_unit_exponent: -6``). The auditory chain
takes microvolts, so the outlet declares and carries microvolts, and the
inlet's declared unit and the chain's ``source_unit_exponent`` agree at -6 with
nothing silently rescaled in between. A volts-declaring publisher would have to
declare `0` at the inlet instead; both are consistent, and the choice is
recorded in the run record rather than left implicit.

Time. The publisher stamps each block with the trial's **own** EEG timestamps on
the LSL/local clock, anchored so that the first published sample is now. The
inlet therefore sees the block's age directly, and ``Acquire.max_lag`` measures
this process's pacing rather than a guess. Nothing here invents a timestamp the
recording does not have.

    python -B -m scripts.getlive.ant_publish --session datasets/AAD-ANT/session_19-34-06.npz \
        --seconds 130
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Where a converted trial records the microvolts the chain consumes.
TRIAL_UNIT = "microvolts"
UNIT_EXPONENT = -6

DEFAULT_STREAM_NAME = "NOVA-ANT-replay"
DEFAULT_SOURCE_ID = "ant-replay-1"
STREAM_TYPE = "EEG"

PUBLISHER_MODULE = "scripts.getlive.ant_publish"
"""Module path a runner invokes to start this publisher as its own process."""


@dataclass(frozen=True)
class AntTrial:
    """One imported ANT session, ready to publish.

    Attributes:
        eeg: ``(samples, channels)`` microvolts.
        timestamps: Source-clock seconds, one per row.
        channel_names: Labels in column order, from the import's own record.
        sample_rate: Source rate in Hz, measured from the timestamps.
        reference: The recording's reference, carried verbatim into the chain.
        upstream_processing: Provenance string from the import.
        labels: Marker-derived per-sample labels; **never** read by the chain.
        audio: The two candidate reference envelopes the import interpolated onto
            this trial's clock, zero where nothing was playing. Not a waveform
            (decision D-33); its first non-zero row is what the session's audio
            anchor is derived from.
        audio_rate: The envelope columns' own rate in Hz, from the import's
            record. Not derived: a recording whose length is not a whole number
            of seconds must not move the anchor.
        subject: Subject id recorded by the import.
        trial_id: Trial id recorded by the import.
    """

    eeg: np.ndarray
    timestamps: np.ndarray
    channel_names: tuple[str, ...]
    sample_rate: float
    reference: str
    upstream_processing: str
    labels: np.ndarray
    audio: np.ndarray
    audio_rate: float
    subject: str
    trial_id: str

    @property
    def seconds(self) -> float:
        """Recording duration in seconds."""

        return float(self.timestamps[-1] - self.timestamps[0])


def load_ant_trial(path: str | Path) -> AntTrial:
    """Load one ``datasets/AAD-ANT/session_*.npz`` trial and check its shape.

    The import writes the channel contract inside the ``metadata`` JSON string
    rather than as its own array, so the labels are read from there - a trial
    without them cannot be published under a named contract and is refused.

    Raises:
        FileNotFoundError: If the trial file is absent.
        ValueError: If the file is missing a field the source contract needs, or
            its shapes disagree.
    """

    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(
            f"ANT trial not found: {target}. Produce it with "
            "`python -B -m scripts.auditory.antneuro --data-root tmp/antneurodata "
            "--out datasets/AAD-ANT`."
        )
    with np.load(target, allow_pickle=False) as archive:
        missing = [
            key
            for key in ("eeg", "timestamps", "labels", "audio", "metadata")
            if key not in archive
        ]
        if missing:
            raise ValueError(f"{target.name} is missing field(s): {', '.join(missing)}")
        eeg = np.asarray(archive["eeg"], dtype=np.float64)
        timestamps = np.asarray(archive["timestamps"], dtype=np.float64)
        labels = np.asarray(archive["labels"], dtype=np.int64)
        audio = np.asarray(archive["audio"], dtype=np.float64)
        metadata = json.loads(str(archive["metadata"]))

    if eeg.ndim != 2:
        raise ValueError(f"{target.name}: eeg must be (samples, channels).")
    if len(timestamps) != eeg.shape[0] or len(labels) != eeg.shape[0]:
        raise ValueError(f"{target.name}: eeg, timestamps and labels disagree in length.")
    names = tuple(str(name) for name in metadata.get("channel_names") or ())
    if len(names) != eeg.shape[1]:
        raise ValueError(
            f"{target.name}: {len(names)} recorded channel names for "
            f"{eeg.shape[1]} columns."
        )
    if eeg.shape[0] < 2:
        raise ValueError(f"{target.name}: too few samples to publish.")
    audio_rate = float(metadata.get("audio_rate") or 0.0)
    if audio_rate <= 0:
        raise ValueError(f"{target.name}: the import records no audio_rate.")
    expected_audio = int(round((timestamps[-1] - timestamps[0]) * audio_rate)) + 1
    if audio.ndim != 2 or abs(audio.shape[0] - expected_audio) > 1:
        raise ValueError(
            f"{target.name}: the audio columns ({audio.shape}) are not at the "
            f"import's {audio_rate:g} Hz against {eeg.shape[0]} EEG samples."
        )
    steps = np.diff(timestamps)
    if not np.all(steps > 0):
        raise ValueError(f"{target.name}: timestamps are not strictly increasing.")
    rate = 1.0 / float(np.median(steps))
    if not np.isclose(rate, round(rate), rtol=0, atol=1e-6):
        raise ValueError(f"{target.name}: implied rate {rate!r} is not a whole number.")
    if not np.all(np.isfinite(eeg)):
        raise ValueError(f"{target.name}: the recording carries non-finite samples.")
    return AntTrial(
        eeg=eeg,
        timestamps=timestamps,
        channel_names=names,
        sample_rate=float(round(rate)),
        reference=str(metadata.get("reference", "not recorded")),
        upstream_processing=str(metadata.get("upstream_processing", "not recorded")),
        labels=labels,
        audio=audio,
        audio_rate=audio_rate,
        subject=str(metadata.get("subject", "ANT")),
        trial_id=str(metadata.get("trial_id", target.stem)),
    )


def slice_window(
    trial: AntTrial, start_seconds: float, seconds: float | None
) -> tuple[AntTrial, int]:
    """Return the trial restricted to ``[start, start + seconds)``, plus the index.

    The returned trial keeps the recording's own timestamps, so a published
    slice is still on the recording's clock and a listener can say where in the
    session it is. Labels travel with the samples: they are scoring material.
    """

    first = int(round(start_seconds * trial.sample_rate))
    if first < 0 or first >= len(trial.timestamps):
        raise ValueError(
            f"--start {start_seconds:g}s is outside the {trial.seconds:.3f}s recording."
        )
    if seconds is None:
        last = len(trial.timestamps)
    else:
        if seconds <= 0:
            raise ValueError("--seconds must be positive.")
        last = min(len(trial.timestamps), first + int(round(seconds * trial.sample_rate)))
    # The audio columns sit at the import's 64 Hz against the 500 Hz EEG, so the
    # same window is the same fraction of both; the integer division is exact
    # because 500/64 = 125/16 and every published window is a whole number of
    # envelope rows in the importer's own grid.
    first_audio = int(round(first * len(trial.audio) / len(trial.timestamps)))
    last_audio = int(round(last * len(trial.audio) / len(trial.timestamps)))
    return (
        AntTrial(
            eeg=trial.eeg[first:last],
            timestamps=trial.timestamps[first:last],
            channel_names=trial.channel_names,
            sample_rate=trial.sample_rate,
            reference=trial.reference,
            upstream_processing=trial.upstream_processing,
            labels=trial.labels[first:last],
            audio=trial.audio[first_audio:last_audio],
            audio_rate=trial.audio_rate,
            subject=trial.subject,
            trial_id=trial.trial_id,
        ),
        first,
    )


def stream_info(trial: AntTrial, *, name: str, source_id: str):
    """Declare the outlet: the name, the rate, and the unit the samples are in.

    Imported lazily so this module can be imported, and its loader tested,
    without MNE-LSL present.
    """

    from mne_lsl.lsl import StreamInfo

    info = StreamInfo(name, STREAM_TYPE, len(trial.channel_names), trial.sample_rate, "float32", source_id)
    info.set_channel_names(list(trial.channel_names))
    info.set_channel_types(["eeg"] * len(trial.channel_names))
    info.set_channel_units([TRIAL_UNIT] * len(trial.channel_names))
    return info


def publish(
    trial: AntTrial,
    *,
    name: str = DEFAULT_STREAM_NAME,
    source_id: str = DEFAULT_SOURCE_ID,
    chunk_samples: int = 25,
    lead_seconds: float = 1.0,
    sleep=time.sleep,
    clock=None,
) -> dict:
    """Publish the trial at its own rate until it ends; return what was sent.

    Pacing is absolute: each block is due at ``first_stamp + block_end_offset``,
    so a slow consumer catches up within the block rate instead of drifting for
    the length of the recording.

    Args:
        trial: What to publish.
        name: Outlet name.
        source_id: Outlet source id.
        chunk_samples: Samples per push; 25 at 500 Hz is a 50 ms block.
        lead_seconds: Head start before the first block, so an inlet in another
            process can attach.
        sleep: Injectable pacing sleep, for tests.
        clock: Injectable monotonic clock, for tests; defaults to
            ``mne_lsl.lsl.local_clock``, the clock the stamps live on.

    Returns:
        A JSON-safe report of what was published.
    """

    from mne_lsl.lsl import StreamOutlet, local_clock

    if chunk_samples < 1:
        raise ValueError("chunk_samples must be positive.")
    if clock is None:
        clock = local_clock

    info = stream_info(trial, name=name, source_id=source_id)
    outlet = StreamOutlet(info, chunk_size=chunk_samples)
    rate = trial.sample_rate
    step = 1.0 / rate
    first = float(clock()) + float(lead_seconds)
    blocks = 0
    sent = 0
    try:
        while sent < len(trial.timestamps):
            count = min(chunk_samples, len(trial.timestamps) - sent)
            rows = trial.eeg[sent : sent + count]
            # The recording's own timestamps, moved onto the clock the inlet
            # reads. Block i's stamp is the stamp of its first sample, which is
            # what liblsl's array form means.
            stamps = first + (trial.timestamps[sent : sent + count] - trial.timestamps[0])
            outlet.push_chunk(np.ascontiguousarray(rows, dtype=np.float32), stamps)
            blocks += 1
            sent += count
            due = first + (float(trial.timestamps[sent - 1]) - trial.timestamps[0]) + step
            delay = due - float(clock())
            if delay > 0:
                sleep(delay)
    finally:
        # Dropping the outlet is what tells an inlet the source has ended.
        del outlet
    return {
        "stream_name": name,
        "source_id": source_id,
        "channels": list(trial.channel_names),
        "sample_rate": trial.sample_rate,
        "unit": TRIAL_UNIT,
        "unit_exponent": UNIT_EXPONENT,
        "blocks": blocks,
        "samples": sent,
        "seconds": round(sent / rate, 3),
        "chunk_samples": chunk_samples,
    }


def build_parser() -> argparse.ArgumentParser:
    """The publisher's own command line."""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--session",
        default="datasets/AAD-ANT/session_19-34-06.npz",
        help="imported ANT trial to publish",
    )
    parser.add_argument("--start", type=float, default=0.0, help="first second to publish")
    parser.add_argument(
        "--seconds", type=float, default=None, help="publish only this much (default: all)"
    )
    parser.add_argument("--name", default=DEFAULT_STREAM_NAME)
    parser.add_argument("--source-id", default=DEFAULT_SOURCE_ID)
    parser.add_argument("--chunk-samples", type=int, default=25)
    parser.add_argument("--lead-seconds", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Load, slice, announce and publish until the slice ends."""

    args = build_parser().parse_args(argv)
    trial = load_ant_trial(args.session)
    window, first = slice_window(trial, args.start, args.seconds)
    print(
        json.dumps(
            {
                "session": str(args.session),
                "trial_id": window.trial_id,
                "first_sample_index": first,
                "channels": list(window.channel_names),
                "sample_rate": window.sample_rate,
                "unit": TRIAL_UNIT,
                "first_timestamp": float(window.timestamps[0]),
                "peak_microvolts": float(np.max(np.abs(window.eeg))),
                "marker_labels_present": sorted(
                    {int(value) for value in window.labels.tolist()}
                ),
            }
        ),
        flush=True,
    )
    report = publish(
        window,
        name=args.name,
        source_id=args.source_id,
        chunk_samples=args.chunk_samples,
        lead_seconds=args.lead_seconds,
    )
    print(json.dumps({"published": report}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
