"""Import the recorded ANT Neuro sessions into the repository's trial interchange format.

The recording is the operator's own, made on the rig the live demo will use:
``tmp/antneurodata/audio/Lacroix_Flo2_2026-09-12_<session>.cnt`` (500 Hz, 24 EEG
channels, read with :func:`mne.io.read_raw_ant` - ``read_raw_cnt`` is a NeuroScan
reader and fails on these files with a misleading error) plus the sibling ``.evt``
file, which ``read_raw_ant`` parses into ``raw.annotations``. The measured
properties of the files, and everything the files cannot answer, are recorded in
``results/antneuro_testset_report.md``. The parts this module depends on, and who
owns them:

* **Channel contract.** The trial carries the 20 electrodes shared with the KU
  Leuven cap, in the order fixed by plan section 3.11. Columns are chosen *by
  name*; the four channels the live cap has and KU Leuven does not (``F9 F10 M1
  M2``) are dropped by name, never by column count.
* **Units.** ``raw.get_data()`` is volts, so the values are multiplied by 1e6 to
  reach the microvolts the chain expects, and both the source unit and the
  factor are written into the trial's metadata.
* **Audio timeline** (operator statement, not recoverable from the files).
  Session ``19-34-06`` plays the stimulus from 0 s, session ``19-41-11`` from
  267.0 s, and within a session the ``1004/Start`` marker is the anchor, so
  ``stimulus position = start offset + (EEG time - Start time)``. The candidate
  columns are built from the precomputed reference envelopes
  ``datasets/audio/left_mono.npz`` / ``right_mono.npz`` through
  :func:`~nova2026.auditory.envelopes.load_envelope`, which refuses a missing or
  stale envelope. Those two envelopes sit on the played channels
  (``left_mono.wav`` / ``right_mono.wav`` are bit-identical to the two channels of
  ``dichotic_15min.wav``); the ``*_raw`` material was never played.
* **Labels** (operator's rule). ``1001/Left side`` means attend candidate A,
  ``1006/Custom Annotation`` means attend candidate B. Every switch is surrounded
  by a symmetric buffer of 0.5-1 s that is unknown (``-1``); the default is
  :data:`DEFAULT_BUFFER_SECONDS` and the operator's range is enforced rather than
  silently left undefined.
* **Marker exclusions** are an explicit table keyed by session, code and time
  (:data:`EXCLUSIONS`). An entry that matches no recorded marker is an error: an
  exclusion that quietly stops applying is exactly how a spurious cue turns into
  a label. The disputed ``1006`` at 148.402 s is *kept* and flagged ambiguous
  (:data:`AMBIGUOUS`) because the file evidence says it is the real cue.

Two things this importer deliberately does not do. It does not exclude or repair
the railed channels (``F8`` sits at the amplifier rail for the whole of both
sessions and ``F3`` for 39.8 % of session 1): it measures them into the summary,
because the chain owns its quality policy and dropping electrodes here would
change the channel contract behind the decoder's back. And it does not decide
whether the derived labels are usable - it produces the numbers a human needs to
decide that, including how much of each session survives the buffers.

The trial's ``audio`` columns are the two candidate **reference envelopes on the
trial clock** (64 Hz, band 1-9 Hz, linearly interpolated onto ``t = k / 500``),
not a waveform: the decoding path never reads ``trial.audio`` (it loads these same
envelopes through ``load_envelope``), and the playable stimulus for these sessions
is the original ``dichotic_15min.wav``, whose per-session sample range the summary
records so a renderer can cut it.

    .venv\\Scripts\\python.exe -B -m scripts.auditory.antneuro
    .venv\\Scripts\\python.exe -B -m scripts.auditory.antneuro --buffer-seconds 1.0
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from nova2026.auditory.config import AuditoryConfig
from nova2026.auditory.data import AuditoryTrial, save_trial
from nova2026.auditory.envelopes import load_envelope

DEFAULT_DATA_ROOT = "tmp/antneurodata"
DEFAULT_OUT_DIR = "datasets/AAD-ANT"
DEFAULT_ENVELOPE_DIR = "datasets/audio"
DEFAULT_REVIEW_DIR = "results"

#: The electrodes the live cap and the KU Leuven dataset share, in the contract
#: order of plan section 3.11. This tuple is the trial's ``channel_names``.
SHARED_CHANNEL_NAMES = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "C4",
    "T8", "Cz", "P7", "P3", "Pz", "P4", "P8", "Oz", "O1", "O2",
)

#: Stimulus position at the session's ``1004/Start`` marker. Operator statement:
#: the files record no playback position, so this cannot be measured and is not
#: guessed - a session absent from this table is refused.
SESSION_AUDIO_START_SECONDS = {"19-34-06": 0.0, "19-41-11": 267.0}

#: Candidate A and B reference envelopes, in that order, under ``datasets/audio``.
CANDIDATE_ENVELOPES = ("left_mono.npz", "right_mono.npz")
#: 1001 = attend the left candidate, 1006 = attend the right candidate.
CUE_CANDIDATES = {"1001": 0, "1006": 1}
START_CODE = "1004"
DEFAULT_BUFFER_SECONDS = 0.5
#: The operator stated "0.5-1 s"; outside that range the lab is not theirs.
BUFFER_RANGE_SECONDS = (0.5, 1.0)
UNIT_SCALES = {"V": 1e6, "mV": 1e3, "uV": 1.0, "\u00b5V": 1.0}
EXCLUSION_TOLERANCE_SECONDS = 0.02
RAIL_FRACTION_THRESHOLD = 0.01
RAIL_PEAK_FRACTION = 0.999

EXCLUSIONS = (
    {
        "session": "19-41-11",
        "code": "1007",
        "time": 148.354,
        "reason": "operator-confirmed spurious: a 2.000 s Saying-YES event, not an attention cue",
    },
    {
        "session": "19-41-11",
        "code": "1004",
        "time": 326.692,
        "reason": "operator-confirmed spurious: a second Start anchor pressed while stopping the recording",
    },
)
AMBIGUOUS = (
    {
        "session": "19-41-11",
        "code": "1006",
        "time": 148.402,
        "reason": (
            "operator reported this right cue as an accidental press, 0.048 s after the "
            "Saying-YES event; retained because dropping it would leave 11 right cues "
            "against 12 left and a 25.670 s left stretch, while keeping it keeps the "
            "alternation at 12.848 s and 12.822 s, both mid-range for this session"
        ),
    },
)


@dataclass
class SessionSource:
    """One opened recording in the units the file stores, before any decision.

    ``data`` is in the trial's orientation, samples by channels; MNE hands back
    the transpose, and :func:`read_cnt_session` is where that is undone.
    """

    path: Path
    session: str
    sample_rate: float
    channel_names: tuple
    data: np.ndarray
    unit: str
    annotations: tuple


def session_key_from_path(path):
    """Name the session from the recording's own file name, refusing anything else."""

    match = re.search(r"_(\d{2}-\d{2}-\d{2})\.cnt$", str(path), re.IGNORECASE)
    if match is None:
        raise ValueError(
            f"Cannot name the session from {Path(path).name!r}: expected the recording "
            "to end in _HH-MM-SS.cnt."
        )
    return match.group(1)


def code_and_name(description):
    """Split a ``.evt`` description such as ``1001/Left side`` into code and name."""

    code, _, name = str(description).partition("/")
    return code.strip(), name.strip()


def select_channel_columns(available, expected=SHARED_CHANNEL_NAMES):
    """Map the contract's channel order onto the recording's columns, by name.

    Returns the source column index for each name in ``expected``. A missing or
    duplicated contract channel is an error naming the channel: selecting by
    count instead is how a montage silently shifts under a decoder.
    """

    names = list(available)
    missing = [name for name in expected if name not in names]
    duplicated = [name for name in expected if names.count(name) > 1]
    if missing:
        raise ValueError(
            "The recording is missing contract channel(s) " + ", ".join(missing) + "."
        )
    if duplicated:
        raise ValueError(
            "The recording has duplicate contract channel(s) " + ", ".join(duplicated) + "."
        )
    return [names.index(name) for name in expected]


def to_microvolts(data, source_unit):
    """Convert stored samples to the microvolts the chain expects, or refuse."""

    if source_unit not in UNIT_SCALES:
        raise ValueError(
            f"Unknown source unit {source_unit!r}; known units are "
            + ", ".join(sorted(UNIT_SCALES))
            + ". Scaling by a guessed factor would corrupt every downstream threshold."
        )
    return np.asarray(data, dtype=float) * UNIT_SCALES[source_unit]


def _match(table, session, code, onset):
    for entry in table:
        if (
            entry["session"] == session
            and entry["code"] == code
            and abs(entry["time"] - float(onset)) <= EXCLUSION_TOLERANCE_SECONDS
        ):
            return entry
    return None


def marker_table(annotations, session):
    """One row per recorded marker, with its disposition. Never drops one.

    Returns ``(rows, used_exclusions, used_ambiguous)``. Raises when an entry of
    :data:`EXCLUSIONS` or :data:`AMBIGUOUS` matches no marker in this session, so
    the table cannot rot into a no-op.
    """

    rows = []
    used_exclusions = set()
    used_ambiguous = set()
    anchored = False
    for index, (onset, duration, description) in enumerate(annotations):
        code, name = code_and_name(description)
        candidate = CUE_CANDIDATES.get(code)
        exclusion = _match(EXCLUSIONS, session, code, onset)
        ambiguous = _match(AMBIGUOUS, session, code, onset)
        if exclusion is not None:
            disposition, reason = "excluded", exclusion["reason"]
            used_exclusions.add(exclusion["time"])
        elif candidate is not None:
            if ambiguous is not None:
                disposition, reason = "ambiguous-cue", ambiguous["reason"]
                used_ambiguous.add(ambiguous["time"])
            else:
                disposition, reason = "cue", "attention cue, used for labels"
        elif code == START_CODE and not anchored:
            disposition, reason = "anchor", "first Start marker: the audio timeline's anchor"
            anchored = True
        elif code == START_CODE:
            disposition, reason = (
                "duplicate-anchor",
                "a further Start marker; only the first one anchors the timeline",
            )
        else:
            disposition, reason = "non-cue", "not an attention cue by code"
        rows.append(
            {
                "index": index,
                "onset_seconds": float(onset),
                "duration_seconds": float(duration),
                "code": code,
                "name": name,
                "candidate": candidate,
                "disposition": disposition,
                "reason": reason,
            }
        )
    unmatched = [
        entry
        for entry in EXCLUSIONS
        if entry["session"] == session and entry["time"] not in used_exclusions
    ]
    unmatched += [
        entry
        for entry in AMBIGUOUS
        if entry["session"] == session and entry["time"] not in used_ambiguous
    ]
    if unmatched:
        raise ValueError(
            f"Session {session}: the marker table's explicit entry/entries "
            + ", ".join(f"{e['code']}@{e['time']}" for e in unmatched)
            + " matched no recorded marker. The table and the file have diverged; "
            "fix the table rather than importing with a silently inert exclusion."
        )
    return rows, used_exclusions, used_ambiguous


def cues_from_rows(rows):
    """The attention cues, in time order, as ``(onset, candidate)`` pairs."""

    cues = [
        (row["onset_seconds"], row["candidate"])
        for row in rows
        if row["disposition"] in ("cue", "ambiguous-cue")
    ]
    return sorted(cues)


def alternation_report(cues):
    """Whether the cues alternate and, if not, where they do not."""

    breaks = []
    for index in range(1, len(cues)):
        if cues[index][1] == cues[index - 1][1]:
            breaks.append(
                {
                    "after_seconds": cues[index - 1][0],
                    "at_seconds": cues[index][0],
                    "candidate": cues[index][1],
                }
            )
    return {"ok": not breaks, "breaks": breaks}


def labels_from_cues(times, cues, buffer_seconds=DEFAULT_BUFFER_SECONDS):
    """Label each sample with the attended candidate, or ``-1`` inside a buffer.

    The operator's rule: after a switch the first 0.5-1 s is unknown, and the
    0.5-1 s before the next switch is unknown too, so the buffer is symmetric and
    its endpoints are included in the unknown span. Before the first cue nothing
    has been asked of the participant, and after the last one the last instruction
    stands until the recording ends.
    """

    if not BUFFER_RANGE_SECONDS[0] <= buffer_seconds <= BUFFER_RANGE_SECONDS[1]:
        raise ValueError(
            f"Buffer {buffer_seconds} s is outside the operator's stated range "
            f"{BUFFER_RANGE_SECONDS} s; a different buffer is a different labelling "
            "rule and must be agreed rather than assumed."
        )
    times = np.asarray(times, dtype=float)
    labels = np.full(times.shape, -1, dtype=int)
    if not cues:
        return labels
    onsets = np.asarray([float(onset) for onset, _ in cues], dtype=float)
    candidates = np.asarray([int(candidate) for _, candidate in cues], dtype=int)
    current = np.searchsorted(onsets, times, side="right") - 1
    known = current >= 0
    labels[known] = candidates[current[known]]
    for onset in onsets:
        labels[(times >= onset - buffer_seconds) & (times <= onset + buffer_seconds)] = -1
    return labels


def assigned_segments(times, cues, labels, sample_rate):
    """Seconds each cue actually keeps, cue by cue - the checkable per-switch number."""

    segments = []
    for index, (onset, candidate) in enumerate(cues):
        stop = cues[index + 1][0] if index + 1 < len(cues) else np.inf
        mask = (labels == candidate) & (times > onset) & (times < stop)
        segments.append(
            {
                "cue_index": index,
                "candidate": int(candidate),
                "onset_seconds": float(onset),
                "assigned_seconds": round(float(mask.sum()) / sample_rate, 3),
            }
        )
    return segments


def stimulus_positions(times, start_seconds, start_offset_seconds):
    """Map trial-clock times onto stimulus positions through the Start anchor."""

    return start_offset_seconds + (np.asarray(times, dtype=float) - start_seconds)


def place_envelope(record, positions, played_from=None):
    """Sample a verified envelope at stimulus positions; zero where it has none.

    Returns ``(values, inside)``. A position has no audio when the envelope does
    not reach it *or* when it precedes the played range: the stimulus file is
    longer than what was played, so sampling it at a position before the session's
    start offset would put material the participant never heard into the
    reference. ``inside`` marks the positions that were actually played, so a
    caller can report the coverage instead of presenting zeros as samples.
    """

    grid = np.asarray(record.timestamps, dtype=float).reshape(-1)
    values = np.asarray(record.envelope, dtype=float).reshape(-1)
    positions = np.asarray(positions, dtype=float)
    inside = (positions >= grid[0]) & (positions <= grid[-1])
    if played_from is not None:
        inside &= positions >= float(played_from)
    sampled = np.zeros(positions.shape, dtype=float)
    sampled[inside] = np.interp(positions[inside], grid, values)
    return sampled, inside


def railed_channels(eeg, names, threshold=RAIL_FRACTION_THRESHOLD):
    """Fraction of each channel sitting at the recording's own peak.

    ``eeg`` is in the trial's orientation, samples by channels, and the fraction
    is taken down each column. ``F8`` is at the rail for all of both sessions, so
    a peak amplitude or a variance computed on it measures the amplifier, not the
    brain. The channel is still reported rather than removed: the chain owns the
    quality policy.
    """

    eeg = np.asarray(eeg, dtype=float)
    if eeg.size == 0:
        return []
    peak = float(np.abs(eeg).max())
    if peak <= 0:
        return []
    fractions = (np.abs(eeg) >= RAIL_PEAK_FRACTION * peak).mean(axis=0)
    return [
        {"name": str(name), "railed_fraction": round(float(fraction), 6)}
        for name, fraction in zip(names, fractions)
        if fraction >= threshold
    ]


def _trial_metadata(source, buffer_seconds, start_anchor, start_offset, config):
    """Prose metadata, because the trial format carries text and not free fields."""

    reference = (
        "CPz, declared in the .cnt header's [Basic Channel Data] for all 24 channels "
        "(REF:CPz). CPz is not one of the recorded electrodes, so it is an implicit "
        "reference, and the ground is not in the file. The KU Leuven dataset is "
        "re-referenced to Cz, so the 20-channel contract shared by the two carries a "
        "different reference on each side."
    )
    upstream = (
        f"ANT Neuro recording {Path(source.path).name}, {source.sample_rate:g} Hz, "
        f"{len(source.channel_names)} recorded channels, read with mne.io.read_raw_ant. "
        f"Trial channels are the {len(SHARED_CHANNEL_NAMES)} electrodes shared with the KU "
        "Leuven cap, selected by name in the plan's section 3.11 order; the channels only "
        "the live cap has are dropped by name. Source unit is "
        f"{source.unit!r} (MNE convention), converted to microvolts by x"
        f"{UNIT_SCALES[source.unit]:g}; the chain's source_unit_exponent for these values "
        "is therefore -6. Audio timeline is operator-given: the stimulus starts at "
        f"{start_offset} s at the {START_CODE}/Start marker ({start_anchor:.3f} s of EEG), so "
        "stimulus position = start offset + (EEG time - Start time). The two audio columns "
        f"are the candidate reference envelopes ({CANDIDATE_ENVELOPES[0]} / "
        f"{CANDIDATE_ENVELOPES[1]}, band {config.band} Hz, {config.sample_rate:g} Hz) "
        "linearly interpolated onto this trial's clock, and zero where the played stimulus "
        "does not reach (nothing was playing before the Start marker). They are not a "
        "waveform: the decoding path loads the same envelopes through load_envelope, and "
        "the playable stimulus is the original dichotic_15min.wav. Labels follow the "
        f"operator's rule: a symmetric {buffer_seconds} s buffer around every switch is "
        "unknown (-1) (their stated range is 0.5-1 s). "
        "Marker exclusions are the operator-confirmed table in scripts/auditory/antneuro.py; "
        "the disputed 1006 at 148.402 s is retained and flagged ambiguous."
    )
    return reference, upstream


def build_session(source, envelopes, *, buffer_seconds=DEFAULT_BUFFER_SECONDS, config=None):
    """Turn one opened session into a trial plus the evidence that describes it."""

    config = config or AuditoryConfig()
    session = source.session
    if session not in SESSION_AUDIO_START_SECONDS:
        raise ValueError(
            f"No operator-given audio start offset for session {session!r}. The playback "
            "position is not recorded in the files, so add the offset to "
            "SESSION_AUDIO_START_SECONDS rather than guessing one."
        )
    if len(envelopes) != len(CANDIDATE_ENVELOPES):
        raise ValueError(
            f"Expected {len(CANDIDATE_ENVELOPES)} candidate envelopes, got {len(envelopes)}."
        )
    start_offset = float(SESSION_AUDIO_START_SECONDS[session])
    data = np.asarray(source.data, dtype=float)
    if data.ndim != 2 or data.shape[1] != len(source.channel_names):
        raise ValueError(
            f"Session data has shape {data.shape} but {len(source.channel_names)} channels "
            "are named. The array must be samples by channels; MNE's own orientation is "
            "channels by samples, and selecting by name requires the two to agree."
        )
    columns = select_channel_columns(source.channel_names)
    dropped = [name for name in source.channel_names if name not in SHARED_CHANNEL_NAMES]
    eeg = to_microvolts(data[:, columns], source.unit)
    rate = float(source.sample_rate)
    times = np.arange(eeg.shape[0]) / rate

    rows, _, _ = marker_table(source.annotations, session)
    anchors = [row for row in rows if row["disposition"] == "anchor"]
    if not anchors:
        raise ValueError(f"Session {session}: no {START_CODE}/Start marker to anchor the audio.")
    start_anchor = float(anchors[0]["onset_seconds"])
    cues = cues_from_rows(rows)
    alternation = alternation_report(cues)
    labels = labels_from_cues(times, cues, buffer_seconds)
    conservative = labels_from_cues(times, cues, BUFFER_RANGE_SECONDS[1])

    audio_rate = float(config.sample_rate)
    audio_times = np.arange(int(np.floor(times[-1] * audio_rate)) + 1) / audio_rate
    positions = stimulus_positions(audio_times, start_anchor, start_offset)
    tracks = []
    coverage = None
    for record in envelopes:
        values, inside = place_envelope(record, positions, played_from=start_offset)
        tracks.append(values)
        coverage = inside if coverage is None else (coverage & inside)
    audio = np.column_stack(tracks)

    reference, upstream = _trial_metadata(
        source, buffer_seconds, start_anchor, start_offset, config
    )
    trial = AuditoryTrial(
        eeg,
        times,
        audio,
        audio_rate,
        labels,
        "ANT",
        f"antneuro_{session}",
        SHARED_CHANNEL_NAMES,
        reference,
        upstream,
        group=f"antneuro|{session}",
    )

    # n / rate, the same convention the recording report and MNE use for a
    # duration, so the label budget below adds up to the session length instead of
    # falling one sample interval short of it.
    duration = float(eeg.shape[0]) / rate
    seconds = {key: round(float((labels == key).sum()) / rate, 3) for key in (-1, 0, 1)}
    conservative_seconds = {
        key: round(float((conservative == key).sum()) / rate, 3) for key in (-1, 0, 1)
    }
    summary = {
        "session": session,
        "source_file": str(source.path),
        "shape": [int(eeg.shape[0]), int(eeg.shape[1])],
        "sample_rate_hz": rate,
        "duration_seconds": round(duration, 3),
        "channels_recorded": len(source.channel_names),
        "channel_names": list(SHARED_CHANNEL_NAMES),
        "dropped_channels": dropped,
        "unit": {
            "source_unit": source.unit,
            "trial_unit": "uV",
            "scale_to_microvolts": UNIT_SCALES[source.unit],
            "chain_source_unit_exponent": -6,
            "peak_microvolts": round(float(np.abs(eeg).max()), 3),
        },
        "audio": {
            "start_offset_seconds": start_offset,
            "start_anchor_seconds": round(start_anchor, 6),
            "audio_rate_hz": audio_rate,
            "envelope_sources": list(CANDIDATE_ENVELOPES),
            "samples": int(audio.shape[0]),
            "covered_samples": int(coverage.sum()),
            "covered_seconds": round(float(coverage.sum()) / audio_rate, 3),
            "first_stimulus_position_seconds": round(float(positions[coverage][0]), 3),
            "last_stimulus_position_seconds": round(float(positions[coverage][-1]), 3),
            "playable_source_slice_samples": [
                int(round(start_offset * 48000)),
                int(round((start_offset + duration) * 48000)),
            ],
        },
        "markers": {
            "total": len(rows),
            "cues": len(cues),
            "candidate_a": sum(1 for _, candidate in cues if candidate == 0),
            "candidate_b": sum(1 for _, candidate in cues if candidate == 1),
            "excluded": [row for row in rows if row["disposition"] == "excluded"],
            "ambiguous": [row for row in rows if row["disposition"] == "ambiguous-cue"],
            "non_cues": [row for row in rows if row["disposition"] == "non-cue"],
            "table": rows,
        },
        "labels": {
            "buffer_seconds": buffer_seconds,
            "unknown_seconds": seconds[-1],
            "a_seconds": seconds[0],
            "b_seconds": seconds[1],
            "surviving_seconds": round(seconds[0] + seconds[1], 3),
            "surviving_fraction": round((seconds[0] + seconds[1]) / duration, 4),
            "surviving_seconds_at_buffer_1_0": round(
                conservative_seconds[0] + conservative_seconds[1], 3
            ),
            "segments": assigned_segments(times, cues, labels, rate),
        },
        "alternation": alternation,
        "railed_channels": railed_channels(eeg, SHARED_CHANNEL_NAMES),
        "trial_id": trial.trial_id,
        "group": trial.group,
        "trial_count": 1,
    }
    return trial, summary


def read_cnt_session(path, *, session=None):
    """Open one ``.cnt`` with the ANT reader and return a :class:`SessionSource`."""

    import mne  # imported here: the tests exercise everything else without MNE

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No recording at {path}.")
    raw = mne.io.read_raw_ant(path, preload=True, verbose="ERROR")
    try:
        # MNE returns channels by samples; the trial, and every test fixture here,
        # is samples by channels, so the transpose happens once, at the boundary.
        data = np.asarray(raw.get_data(), dtype=float).T
        source = SessionSource(
            path=path,
            session=session or session_key_from_path(path),
            sample_rate=float(raw.info["sfreq"]),
            channel_names=tuple(raw.ch_names),
            data=data,
            unit="V",
            annotations=tuple(
                (float(onset), float(duration), str(description))
                for onset, duration, description in zip(
                    raw.annotations.onset,
                    raw.annotations.duration,
                    raw.annotations.description,
                )
            ),
        )
    finally:
        raw.close()
    return source


def load_candidate_envelopes(directory, config):
    """Load both candidate envelopes, refusing a missing or unverifiable one."""

    directory = Path(directory)
    return tuple(
        load_envelope(directory / name, config) for name in CANDIDATE_ENVELOPES
    )


def import_session(source, out_dir, envelope_dir=DEFAULT_ENVELOPE_DIR, **kwargs):
    """Build one session's trial and write it with its marker table."""

    out_dir = Path(out_dir)
    config = kwargs.pop("config", None) or AuditoryConfig()
    envelopes = load_candidate_envelopes(envelope_dir, config)
    trial, summary = build_session(source, envelopes, config=config, **kwargs)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / f"session_{source.session}.npz"
    save_trial(trial, npz)
    markers = out_dir / f"session_{source.session}_markers.csv"
    write_marker_csv(summary["markers"]["table"], markers)
    summary["trial_file"] = str(npz)
    summary["marker_table_file"] = str(markers)
    return trial, summary


def write_marker_csv(rows, path):
    """Write the marker table where a human can read it beside the recording."""

    fields = [
        "index",
        "onset_seconds",
        "duration_seconds",
        "code",
        "name",
        "candidate",
        "disposition",
        "reason",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def render_markdown(report):
    """The reviewable summary: what was imported, and what a human should check."""

    lines = [
        "# ANT Neuro import - marker table and per-session summary",
        "",
        f"Generated {report['generated_at']} by `{report['command']}`.",
        f"Buffer: **{report['buffer_seconds']} s** symmetric around every switch "
        "(operator's range 0.5-1.0 s; the conservative end is reported too).",
        "",
        "| Session | Samples x ch | Rate | Duration | -1 | A | B | Survives | "
        "Survives @1.0s | Cues (A/B) | Trials |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for summary in report["sessions"]:
        labels = summary["labels"]
        lines.append(
            f"| {summary['session']} | {summary['shape'][0]} x {summary['shape'][1]} | "
            f"{summary['sample_rate_hz']:g} Hz | {summary['duration_seconds']:.2f} s | "
            f"{labels['unknown_seconds']:.1f} s | {labels['a_seconds']:.1f} s | "
            f"{labels['b_seconds']:.1f} s | {labels['surviving_fraction']:.3f} | "
            f"{labels['surviving_seconds_at_buffer_1_0']:.1f} s | "
            f"{summary['markers']['candidate_a']}/{summary['markers']['candidate_b']} | "
            f"{summary['trial_count']} |"
        )
    for summary in report["sessions"]:
        lines += ["", f"## Session {summary['session']}", ""]
        audio = summary["audio"]
        lines += [
            f"- File: `{summary['source_file']}`",
            f"- Channels: {summary['channels_recorded']} recorded -> "
            f"{len(summary['channel_names'])} contract channels "
            f"({', '.join(summary['channel_names'])}); dropped by name: "
            f"{', '.join(summary['dropped_channels'])}",
            f"- Units: source `{summary['unit']['source_unit']}` -> trial "
            f"`{summary['unit']['trial_unit']}` (x{summary['unit']['scale_to_microvolts']:g}); "
            f"peak {summary['unit']['peak_microvolts']:.0f} uV "
            "(the peak is the railed channel, not EEG)",
            f"- Audio: start offset {audio['start_offset_seconds']} s at the Start marker "
            f"{audio['start_anchor_seconds']:.3f} s; stimulus "
            f"{audio['first_stimulus_position_seconds']:.1f}-"
            f"{audio['last_stimulus_position_seconds']:.1f} s of the played file; "
            f"{audio['covered_seconds']:.1f} s of {summary['duration_seconds']:.1f} s covered",
            f"- Alternation: {'strict' if summary['alternation']['ok'] else 'BROKEN'}",
            f"- Railed channels (measured, not removed): "
            + (
                ", ".join(
                    f"{row['name']} {100 * row['railed_fraction']:.2f} %"
                    for row in summary["railed_channels"]
                )
                or "none"
            ),
            "",
            "Markers, as recorded (every one appears here; none is dropped silently):",
            "",
            "| # | EEG t (s) | dur (s) | code | name | candidate | disposition | reason |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in summary["markers"]["table"]:
            candidate = (
                "A" if row["candidate"] == 0 else "B" if row["candidate"] == 1 else "-"
            )
            lines.append(
                f"| {row['index']} | {row['onset_seconds']:.3f} | "
                f"{row['duration_seconds']:.3f} | {row['code']} | {row['name']} | "
                f"{candidate} | {row['disposition']} | {row['reason']} |"
            )
    lines += ["", "## What to check by hand", ""]
    lines += [f"- {item}" for item in report["checks"]]
    lines += ["", "## Notes and limits", ""]
    lines += [f"- {item}" for item in report["notes"]]
    return "\n".join(lines) + "\n"


def notes_for_report():
    """The limits a reader must not have to infer from the numbers."""

    return [
        "The trial's audio columns are the two candidate reference envelopes on the trial "
        "clock (64 Hz, band 1-9 Hz), not a waveform; the decoding path loads the same "
        "envelopes through load_envelope, and the playable stimulus is the original "
        "dichotic_15min.wav (see playable_source_slice_samples for the per-session range).",
        "The audio start offsets (0 s and 267.0 s) are operator statements. Nothing in the "
        "recorded files can confirm them, so an error there would shift every window.",
        "The reference is CPz and the ground is not in the file, while the KU Leuven trials "
        "are Cz-referenced. A decoder trained on one set is not reference-matched to the other.",
        "F8 is at the amplifier rail for the whole of both sessions and F3 for 39.8 % of "
        "session 1. They are kept in the 20-channel contract and reported here; no amplitude, "
        "variance or band-power statement may be made about them.",
        "This recording is dichotic (one candidate per ear), so it is a rehearsal of the "
        "task, not the same-mixture presentation the demo aims at.",
        "One continuous session is one trial: the switching lives in the labels, not in trial "
        "boundaries, so a window-level evaluation must cut windows inside the labelled spans.",
    ]


def checks_for_report():
    """The concrete hand checks that would falsify this import."""

    return [
        "Open the marker table beside the .evt file (or the ANT software's event list) and "
        "confirm every onset, code and name, including the ones marked excluded.",
        "Confirm the two exclusions are the operator's own list: the 2.000 s 1007/Saying-YES "
        "and the second 1004/Start.",
        "Confirm the disputed 1006 at 148.402 s. It is retained, flagged ambiguous; the "
        "marker table shows the arithmetic that keeps it (12 right cues, alternation intact).",
        "Listen to the first seconds of session 2's recorded stimulus near 267 s and confirm "
        "the anchor: the audio at the Start marker must match what was playing then.",
        "Check the label distribution: a candidate's seconds should be its own, minus the "
        "buffers around its switch; per-cue 'assigned_seconds' shows where a switch kept "
        "nothing, which is what happens when two cues fall closer than twice the buffer.",
        "Decide whether the labels are usable at all: check that the surviving fraction is "
        "high enough, and that the railed channels do not leave the 20-channel contract with "
        "fewer usable electrodes than the decoder was trained on.",
    ]


def collect_report(summaries, *, buffer_seconds, command, stamp):
    """Assemble the committed review document."""

    return {
        "kind": "step-9.5 ANT Neuro import: marker table and per-session summary",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stamp": stamp,
        "command": command,
        "buffer_seconds": buffer_seconds,
        "buffer_range_seconds": list(BUFFER_RANGE_SECONDS),
        "session_audio_start_seconds": dict(SESSION_AUDIO_START_SECONDS),
        "channel_contract": list(SHARED_CHANNEL_NAMES),
        "exclusions": [dict(entry) for entry in EXCLUSIONS],
        "ambiguous": [dict(entry) for entry in AMBIGUOUS],
        "sessions": summaries,
        "checks": checks_for_report(),
        "notes": notes_for_report(),
        "total_trials": sum(summary["trial_count"] for summary in summaries),
    }


def discover_sessions(data_root):
    """Every ``.cnt`` under the recording root, named by its own file name."""

    audio = Path(data_root) / "audio"
    found = sorted(audio.glob("*.cnt"))
    if not found:
        raise FileNotFoundError(
            f"No .cnt recordings under {audio}. The test recording is not committed; "
            "point --data-root at a copy of tmp/antneurodata."
        )
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description="Import the ANT Neuro sessions.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out", default=DEFAULT_OUT_DIR)
    parser.add_argument("--envelopes", default=DEFAULT_ENVELOPE_DIR)
    parser.add_argument("--review-dir", default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--buffer-seconds", type=float, default=DEFAULT_BUFFER_SECONDS)
    parser.add_argument("--stamp", default=None, help="defaults to the current UTC time")
    args = parser.parse_args(argv)

    stamp = args.stamp or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    config = AuditoryConfig()
    command = (
        f"python -B -m scripts.auditory.antneuro --data-root {args.data_root} "
        f"--out {args.out} --buffer-seconds {args.buffer_seconds}"
    )
    summaries = []
    for path in discover_sessions(args.data_root):
        source = read_cnt_session(path)
        _, summary = import_session(
            source,
            args.out,
            args.envelopes,
            buffer_seconds=args.buffer_seconds,
            config=config,
        )
        summaries.append(summary)
        print(
            f"{summary['session']}: {summary['shape'][0]} x {summary['shape'][1]} @ "
            f"{summary['sample_rate_hz']:g} Hz, {summary['duration_seconds']:.2f} s, "
            f"{summary['markers']['cues']} cues, -1 {summary['labels']['unknown_seconds']:.1f} s, "
            f"A {summary['labels']['a_seconds']:.1f} s, B {summary['labels']['b_seconds']:.1f} s, "
            f"survives {100 * summary['labels']['surviving_fraction']:.1f} %"
        )
    report = collect_report(
        summaries, buffer_seconds=args.buffer_seconds, command=command, stamp=stamp
    )
    review_dir = Path(args.review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)
    json_path = review_dir / f"antneuro_import_{stamp}.json"
    md_path = review_dir / f"antneuro_import_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    summary_path = Path(args.out) / "import_summary.json"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"trials written under {args.out}")
    print(f"review: {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
