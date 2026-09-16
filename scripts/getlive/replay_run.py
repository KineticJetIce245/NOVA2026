"""Replay a recorded getlive run through the live chain, without an amplifier.

Run from the repository root:

    .venv/bin/python -B -m scripts.getlive.replay_run records/nova2026/bringup30/run-20260912-172216
    .venv/bin/python -B -m scripts.getlive.replay_run <run-dir> --timebase stamps
    .venv/bin/python -B -m scripts.getlive.replay_run <run-dir> --out /tmp/replay.json

A getlive run can only be measured with the amplifier in front of you, and the
runs kept in ``records/`` outlive it. This tool feeds a recording back through
**the live script's own chain and its own acceptance rules** - ``live._prepare``,
``live._loop`` and ``live._finish``, with acquisition swapped for the recorded
blocks - so "would the current code have handled that session?" can be answered
without the cap. The samples and the chunk anchors are the recorded ones; the one
thing a replay cannot have is a transport, and the two transport rules say so
rather than printing a zero.

Three consequences of that design are worth stating:

* The replay's clock is the recording's own timeline, so a 30 s recording runs
  30 s of stream time and stops at its end. Nothing here depends on the wall
  clock, which also makes a replay deterministic.
* A recording holds the columns the run *contracted*, so the outlet's original
  montage is not re-checked: the replay answers what the chain did with this
  data, not what the amplifier published.
* Options default to what the recording itself used, so a replay reproduces the
  session; the flags exist to change the decision under test - above all
  ``--timebase``, whose recorded value is printed for comparison.

``scripts/verify_realdata.py`` is the other replayer and answers a different
question: it pushes a public dataset through the demo's chain to check that
offline processing and the live chain agree.
"""

import argparse
from pathlib import Path
from unittest import mock

import numpy as np

from nova2026.streaming.preflight import ChannelContract
from nova2026.streaming.recording import read_metadata, replay_chunks

from . import live

REPLAY_ROLE = "replay"

# The chain limits a replay may change, with the flag each one maps to. Their
# defaults are the live script's, read from its parser so the numbers have one
# owner.
_LIMITS = (
    ("--repair-amplitude-uv", float, "repair_amplitude_uv",
     "largest jump a repair may bridge, in uV"),
    ("--repair-saturation-uv", float, "repair_saturation_uv",
     "absolute level that makes a repair unsafe, in uV"),
    ("--max-recoveries", int, "max_recoveries",
     "chain restarts allowed before the run stops"),
    ("--max-fault-seconds", float, "max_fault_seconds",
     "consecutive judge-rejected windows tolerated"),
)

_UNITS_BY_EXPONENT = {exponent: name for name, exponent in live.SOURCE_UNITS.items()}


def _add_chain_args(parser: argparse.ArgumentParser) -> None:
    """Register the chain decisions a replay may change."""

    parser.add_argument(
        "--timebase",
        choices=live.TIME_BASE_MODES,
        default="grid",
        help="grid (default, and what the live script does now): rebuild the "
        "timeline from a counted index; stamps: keep the recorded stamps, which "
        "is what a run recorded before the default moved did",
    )
    parser.add_argument(
        "--exclude-channels",
        default=None,
        help="EEG labels kept out of the fault verdict (default: the recording's "
        "own list; pass an empty string to clear it)",
    )
    parser.add_argument(
        "--no-channel-check",
        dest="no_channel_check",
        action="store_true",
        default=None,
        help="record channel faults but never let them reject a window "
        "(default: as recorded)",
    )
    parser.add_argument(
        "--channel-check",
        dest="no_channel_check",
        action="store_false",
        help="the opposite, for a recording that was made with the check off",
    )
    parser.add_argument(
        "--out-sfreq",
        type=float,
        default=None,
        help="output rate after resampling (default: as recorded)",
    )


def _add_limit_args(parser: argparse.ArgumentParser) -> None:
    """Register the fault limits, unset meaning "what the live script uses".

    The live script defaults these to numbers; a replay leaves them unset so the
    recording's own chain is reproduced unless the operator asks otherwise.
    """

    for flag, kind, dest, meaning in _LIMITS:
        parser.add_argument(
            flag,
            type=kind,
            default=None,
            help=f"{meaning}; the live script's default is {_live_default(dest):g}",
        )


def _add_output_args(parser: argparse.ArgumentParser) -> None:
    """Register where the replay reports and re-records itself."""

    parser.add_argument("--out", type=Path, help="write the acceptance report as JSON")
    parser.add_argument(
        "--record", type=Path, help="re-record the replayed run under this root"
    )
    parser.add_argument("--subject", help="subject id when re-recording")
    parser.add_argument("--session", help="session id when re-recording")
    parser.add_argument("--run", help="run name when re-recording")
    parser.add_argument(
        "--progress",
        action="store_true",
        help="print one line per window; its lag column measures the wall clock "
        "against the recorded stamps, so it means nothing offline",
    )


def build_parser() -> argparse.ArgumentParser:
    """The chain decisions a replay may change, and nothing else.

    Everything else comes from the recording or from the live script's own
    defaults. Options a replay cannot honour are not accepted at all, rather
    than accepted and ignored.
    """

    parser = argparse.ArgumentParser(
        prog="python -m scripts.getlive.replay_run",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Named ``recording`` on purpose: the live parser already owns ``--run``,
    # and a positional sharing that dest would let ``--run`` overwrite the path.
    parser.add_argument(
        "recording",
        type=Path,
        metavar="RUN",
        help="the recording directory, or its .sqlite/.json file",
    )
    _add_chain_args(parser)
    _add_limit_args(parser)
    _add_output_args(parser)
    return parser


def _live_default(dest: str):
    """The live script's own default for one option."""

    return live.build_parser().get_default(dest)


def find_database(path: Path) -> Path:
    """The run's SQLite file, from a directory or from either of its siblings."""

    if path.is_dir():
        found = sorted(p for p in path.glob("*.sqlite") if not p.name.startswith("._"))
        if len(found) != 1:
            raise SystemExit(
                f"{path}: expected exactly one .sqlite recording, found {len(found)}"
            )
        return found[0]
    sibling = path.with_suffix(".sqlite")
    if sibling.is_file():
        return sibling
    raise SystemExit(f"{path}: not a recorded run (no .sqlite beside it)")


def load(path: Path) -> tuple[Path, dict, list[tuple[np.ndarray, np.ndarray]]]:
    """The recording's database, metadata and chunks with rebuilt timestamps."""

    database = find_database(path)
    metadata = read_metadata(database)
    if float(metadata.get("sfreq") or 0.0) <= 0:
        raise SystemExit(f"{database}: the recording does not say what rate it used")
    if not metadata.get("channels"):
        raise SystemExit(f"{database}: the recording does not say what channels it has")
    chunks = [(data, times) for data, times in replay_chunks(database) if times.size]
    if not chunks:
        raise SystemExit(f"{database}: the recording holds no samples")
    return database, metadata, chunks


def unit_name(exponent: int) -> str:
    """The live script's name for a unit exponent, or a loud failure."""

    try:
        return _UNITS_BY_EXPONENT[exponent]
    except KeyError:
        raise SystemExit(
            f"the recording declares unit exponent {exponent}, which the live "
            "script has no name for"
        ) from None


def _source(metadata: dict) -> dict:
    """The outlet facts the run recorded, in the shape ``live`` expects."""

    channels = list(metadata["channels"])
    types = list(metadata.get("ch_types") or ["eeg"] * len(channels))
    unit = unit_name(int(metadata.get("unit_exponent", -6)))
    return {
        "name": f"replay:{metadata.get('run', 'recording')}",
        "stype": "EEG",
        "source_id": "",
        "n_channels": len(channels),
        "sfreq": float(metadata["sfreq"]),
        "channels": channels,
        "types": types,
        "units": [unit] * len(channels),
    }


def recorded_span(chunks) -> float:
    """Stream time the recording covers, from its first to its last stamp."""

    return float(chunks[-1][1][-1]) - float(chunks[0][1][0])


def replay_args(metadata: dict, args: argparse.Namespace, span: float) -> argparse.Namespace:
    """The live script's argument surface, seeded from the recording.

    Live's parser supplies every chain option at its own default; this then says
    what the recording itself used, and finally what the operator asked for.
    """

    config = metadata.get("config") or {}
    policy = config.get("channel_policy") or {}
    sfreq = float(metadata["sfreq"])
    merged = live.build_parser().parse_args(
        ["--sfreq", f"{sfreq:g}", "--source-units", unit_name(int(metadata.get("unit_exponent", -6)))]
    )
    _as_recorded(merged, config, policy, args)
    merged.duration = span
    # A replay always trusts the recording's own channel list, so demanding a
    # datasheet profile would have nothing to check it against.
    merged.cap = "declared"
    merged.role = REPLAY_ROLE
    merged.quiet = not args.progress
    # What the operator asked for, last so it wins.
    merged.timebase = args.timebase
    merged.out = args.out
    merged.record = args.record
    merged.subject = args.subject or metadata.get("subject")
    merged.session = args.session or metadata.get("session")
    merged.run = args.run
    for _flag, _kind, dest, _meaning in _LIMITS:
        value = getattr(args, dest)
        if value is not None:
            setattr(merged, dest, value)
    # The replay marks its run: a replay has no transport, and the two transport
    # rules then report that instead of scoring a counter nobody produced.
    merged.replay = True
    return merged


def _as_recorded(merged, config: dict, policy: dict, args: argparse.Namespace) -> None:
    """Put the contract and quality policy the recording used onto the namespace.

    Anything the operator stated on the command line wins; everything else is
    what the session itself ran with, so a replay reproduces the run.
    """

    merged.channels = config.get("channel_mode", merged.channels)
    merged.eog = "eog" if config.get("eog_channels") else "drop"
    merged.window = config.get("window_seconds", merged.window)
    merged.hop = config.get("step_seconds", merged.hop)
    merged.warmup = config.get("warmup_seconds", merged.warmup)
    merged.out_sfreq = config.get("out_sfreq", merged.out_sfreq)
    merged.max_bad_channels = policy.get("max_bad_channels", merged.max_bad_channels)
    if args.exclude_channels is not None:
        merged.exclude_channels = args.exclude_channels
    else:
        merged.exclude_channels = ",".join(policy.get("excluded_channels") or ())
    if args.no_channel_check is not None:
        merged.no_channel_check = args.no_channel_check
    else:
        merged.no_channel_check = not policy.get("check_channels", True)


class ReplayStream:
    """Enough of ``StreamLSL`` for ``StreamSession`` to be built around one.

    The session reads ``ch_names`` and ``info['nchan']`` while it is assembled.
    Nothing here connects to anything: every sample comes from the recording,
    through the acquire object swapped in below.
    """

    def __init__(self, channels) -> None:
        self.ch_names = list(channels)
        self.info = {"nchan": len(channels)}
        self.callbacks = []
        self.connected = False
        self.n_new_samples = 0

    def add_callback(self, callback) -> None:
        """Accept a callback the real stream would use, and never fire it."""

        self.callbacks.append(callback)

    def acquire(self):
        """Nothing: the replay never asks the session for samples."""

        return None


class ReplayAcquire:
    """``live._loop``'s acquisition surface, fed from the recording.

    The clock is the recording's own timeline: the loop reads it through
    ``monotonic`` to decide when the run is over, so a replay ends where the
    recording ends instead of after some wall-clock duration.
    """

    def __init__(self, chunks) -> None:
        self._chunks = chunks
        self._next = 0
        self._origin: float | None = None
        self._elapsed = 0.0
        self.reads = 0
        # A replay has no transport, so these are never measured; the acceptance
        # rules say so rather than read them.
        self.pending_samples = 0
        self.max_lag = 0.0
        self.gaps = 0
        self.closed = False

    def clock(self) -> float:
        """Stream time the replay has reached, in seconds."""

        return self._elapsed

    def read(self, timeout=None):
        """Hand out the next recorded block, or say the recording ran out."""

        if self._next >= len(self._chunks):
            raise SystemExit(
                f"the recording ran out after {self.reads} block(s); "
                "its timeline is shorter than the run was asked for"
            )
        data, timestamps = self._chunks[self._next]
        self._next += 1
        self.reads += 1
        if self._origin is None:
            self._origin = float(timestamps[0])
        self._elapsed = float(timestamps[-1]) - self._origin
        return data.copy(), timestamps.copy()

    def close(self) -> None:
        """Stop pretending to hold a connection."""

        self.closed = True


def format_replay(database: Path, metadata: dict, args, profile, eeg, excluded) -> str:
    """What is being replayed, and the one thing a replay cannot measure."""

    config = metadata.get("config") or {}
    recorded = (config.get("timebase") or {}).get("mode", "stamps (before the default moved)")
    return "\n".join(
        (
            f"replay    : {database}",
            f"recording : status={metadata.get('status')} role={metadata.get('role')} "
            f"run={metadata.get('run')}",
            f"source    : {metadata['sfreq']:g} Hz, {len(metadata['channels'])} channels, "
            f"unit exponent {metadata.get('unit_exponent')} "
            f"({unit_name(int(metadata.get('unit_exponent', -6)))})",
            f"contract  : {profile.name} profile, {len(eeg)} EEG, "
            f"excluded={list(excluded) or 'none'}, "
            f"channel check {'off' if args.no_channel_check else 'on'}",
            f"timebase  : {args.timebase} (recorded with: {recorded})",
            "note      : a replay has no LSL transport, so input_lag and "
            "timing_gaps report \"not measured\".",
            "",
        )
    )


def replay(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Build the run around the recording, stream it and score it."""

    database, metadata, chunks = load(args.recording)
    span = recorded_span(chunks)
    if span <= 0:
        parser.error(f"{database}: the recording covers no stream time")
    run_args = replay_args(metadata, args, span)
    source = _source(metadata)
    resolved = live._resolve_cap(parser, run_args, source)
    if resolved is None:
        return 1
    profile, eeg, eog, excluded = resolved
    print(
        format_replay(database, metadata, run_args, profile, eeg, excluded),
        end="",
    )
    run = live._Run(
        args=run_args,
        stream=ReplayStream(source["channels"]),
        source=source,
        contract=ChannelContract(tuple(source["channels"]), eeg + eog),
        profile=profile,
        eeg=eeg,
        eog=eog,
        excluded=excluded,
    )
    acquire = ReplayAcquire(chunks)
    # The loop's only clock is ``live.monotonic``; pointing it at the recording's
    # own timeline is what makes the run end at the end of the data.
    with mock.patch.object(live, "monotonic", new=acquire.clock):
        live._prepare(run)
        run.session.acquire = acquire
        live._loop(run)
        return live._finish(run)


def main(argv: list[str] | None = None) -> int:
    """Replay one recorded run and print the verdict it would get today."""

    parser = build_parser()
    return replay(parser, parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
