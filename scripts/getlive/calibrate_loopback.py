"""Measure the audio-to-EEG loopback offset. Plan sections 3.11 / 3.17-4.

    .venv/Scripts/python.exe -B -m scripts.getlive.calibrate_loopback --procedure
    .venv/Scripts/python.exe -B -m scripts.getlive.calibrate_loopback --make-clicks
    .venv/Scripts/python.exe -B -m scripts.getlive.calibrate_loopback --live --method cable

**Why this command has to exist.** The decoder's tolerance is about +-100 ms.
Shifting the audio deliberately against the EEG costs **0.119 (audio early) /
0.150 (audio late) balanced accuracy per 100 ms**, measured in
``results/aad_shift_sweep_*`` - five to seven times the 0.022 the whole
"64 electrodes -> the 20 the headset has" reduction costs. The alignment budget
is written as +-100 ms, and step 11's loopback measurement has a +-30 ms
tolerance. **That number has never been measured on this rig.** Until it is,
nobody can say whether the live demo is aligned, and every decision it publishes
carries an unknown offset inside it.

**What it measures, exactly.** A click train is played through the machine's own
audio output; the amplifier picks the clicks up - electrically through a
loopback cable into a spare electrode input (``--method cable``, the precise
one), or acoustically through headphones (``--method acoustic``, which is a
different and larger quantity: it adds the acoustic path and the auditory
response, so it is *not* the number to compensate with). Each click's position
on the **chain's own time axis** (``Acquire`` -> ``TimeBase`` -> the rebased
session clock the envelopes are indexed on) is compared with the instant the
playback was requested, and the offset is the median of those differences.

The two clocks involved - the LSL stamp clock and this process's monotonic clock
- are mapped by one paired reading taken at the instant playback is requested,
so nothing here assumes they are the same clock.

**What it does not do.** It does not resample, filter or decode anything: it
measures one number. A recording is not required and participant data is not
involved - the loopback electrode carries a cable, not a person.

Exit codes: ``0`` measured, ``2`` refused or not measurable (no amplifier, no
clicks found, or clicks too irregular to trust), ``1`` runtime error.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:  # allow `python scripts/getlive/calibrate_loopback.py`
    sys.path.insert(0, str(REPO))

from scripts.getlive.live_source import (  # noqa: E402
    DEFAULT_EXPECTED_CHANNELS,
    DEFAULT_MODEL,
    DEFAULT_SFREQ,
    REFUSAL,
    model_electrodes,
    unit_exponent,
)
from scripts.getlive.outlets import channel_facts, open_inlet, wait_for_outlet  # noqa: E402

DEFAULT_WAV = "output/auditory_ui/loopback_clicks.wav"
DEFAULT_PROFILE = "results/loopback_calibration.json"
DEFAULT_INTERVAL_SECONDS = 2.0
DEFAULT_LEAD_SECONDS = 4.0
DEFAULT_CLICK_MS = 2.0
DEFAULT_AUDIO_RATE = 48000

TOLERANCE_SECONDS = 0.030
"""The plan's own tolerance for the residual after calibration (section 3.17-4)."""

METHODS = ("cable", "acoustic")

PROCEDURE = """\
Audio-to-EEG loopback calibration - what to do, in order
========================================================

Prerequisites
  1. The amplifier is publishing LSL (command 1, --mode direct, must exit 0).
  2. A loopback path exists between the machine's audio output and the
     amplifier: the precise one is a cable from the headphone output into a
     spare electrode input ("--method cable"). Headphones on a head works too
     ("--method acoustic") but measures a larger, different quantity.
  3. The amplifier's gain and sampling rate are the ones the session will use.

Steps
  1. Build the click track and read its schedule:
       .venv/Scripts/python.exe -B -m scripts.getlive.calibrate_loopback --make-clicks
     It prints "click k at t = lead + k*interval s" and writes the WAV.
  2. Connect the loopback. For --method cable: headphone output -> the spare
     electrode input. Turn the output volume to about half; a click that rails
     the amplifier still measures, but a click nobody can see does not.
  3. Measure:
       .venv/Scripts/python.exe -B -m scripts.getlive.calibrate_loopback --live \\
           --method cable --wav output/auditory_ui/loopback_clicks.wav
     It resolves the amplifier, waits for one block so the timeline exists,
     writes the WAV, starts playback through the system's own player, acquires
     for --seconds, finds the clicks, and reports the offset.
  4. Read the verdict. "measured" means the clicks were found and their spacing
     matched the schedule. "not measurable" means they were not - check the
     cable, the volume and the gain before believing anything else.
  5. Use it:
       .venv/Scripts/python.exe -B -m scripts.auditory_ui.live --calibration \\
           results/loopback_calibration.json
     Command 2 records the offset it used; --require-calibration makes a missing
     profile a refusal instead of a warning.

What the number means
  offset = (the instant the click appears on the chain's own time axis)
         - (the instant playback was requested)
  A positive offset means the sound reaches the EEG *later* than the timeline
  says, which is the usual case: the audio device's own output buffer sits in
  front of it. It therefore includes the sound card's output latency, the cable
  and the amplifier, and the LSL delivery. It does not include anything the gain
  gate does.
  With --method acoustic the number additionally includes the acoustic path and
  the auditory evoked response; it is an upper bound, not the compensation
  value, and the command says so in the profile.

What it cannot tell you
  Whether the alignment stays put. One measurement is one measurement: repeat
  it (--repeat 2) and look at the spread the profile records. A spread wider
  than +-30 ms means the system cannot be called aligned on this hardware yet.
"""


def click_track(
    *,
    seconds: float,
    rate: int,
    interval: float = DEFAULT_INTERVAL_SECONDS,
    lead: float = DEFAULT_LEAD_SECONDS,
    click_ms: float = DEFAULT_CLICK_MS,
    amplitude: float = 0.9,
) -> tuple[np.ndarray, tuple[float, ...]]:
    """A click train and the exact times of its clicks, on its own time axis.

    The clicks are alternating in polarity. That matters for the cable method:
    a DC-coupled loopback through an electrode lead charges the input, and an
    alternating train keeps the baseline where it started instead of walking
    away from it over a minute-long run.

    Returns:
        ``(waveform, click_times)`` - float32 mono in [-1, 1], and the click
        instants in seconds from the start of the file.
    """

    if not 0 < lead < seconds:
        raise ValueError("lead must be positive and shorter than the file.")
    if interval <= 0 or click_ms <= 0 or rate <= 0:
        raise ValueError("interval, click_ms and rate must all be positive.")
    if not 0 < amplitude <= 1:
        raise ValueError("amplitude must be in (0, 1].")
    count = int(round(seconds * rate))
    wave = np.zeros(count, dtype=np.float64)
    width = max(int(round(click_ms * 1e-3 * rate)), 1)
    window = np.hanning(width)
    times = []
    index = 0
    position = float(lead)
    while position + click_ms * 1e-3 < seconds:
        start = int(round(position * rate))
        if start + width >= count:
            break
        wave[start : start + width] += (
            amplitude if index % 2 == 0 else -amplitude
        ) * window
        times.append(round(float(position), 9))
        index += 1
        position += float(interval)
    return wave.astype(np.float32), tuple(times)


def write_click_wav(path: Path, wave: np.ndarray, rate: int) -> dict:
    """Write the click train as a stereo int16 WAV, identical in both ears.

    Both ears get the same clicks so the cable can be plugged into either
    channel, and so no per-ear gain can make one side invisible.
    """

    from scipy.io import wavfile

    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(wave, -1.0, 1.0)
    stereo = np.column_stack([clipped, clipped])
    wavfile.write(path, int(rate), (stereo * 32767.0).astype(np.int16))
    return {"path": str(path), "sample_rate": int(rate),
            "seconds": round(wave.shape[0] / rate, 3)}


def onset_signal(eeg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample transient strength, and which channel carried it.

    A click is a step, not a level: the first difference is where it lives, and
    a slow drift - which is what a real electrode does over a minute - has
    almost none. Taking the maximum across channels means the loopback electrode
    does not have to be named in advance.
    """

    data = np.asarray(eeg, dtype=np.float64)
    if data.ndim != 2 or data.shape[0] < 3:
        raise ValueError("onset_signal needs samples by channels, at least three rows.")
    steps = np.abs(np.diff(data, axis=0))
    return steps.max(axis=1), steps.argmax(axis=1)


def detect_click_times(
    eeg: np.ndarray,
    rate: float,
    *,
    channel_names: tuple[str, ...] | None = None,
    mad_multiple: float = 12.0,
    refractory_seconds: float = 0.5,
) -> dict:
    """Find click instants in an acquired window, on the window's own clock.

    The threshold is a robust one - the median plus ``mad_multiple`` scaled
    median absolute deviations of the first-difference signal - so a channel
    that drifts, or a single large artifact, cannot set it. A refractory period
    of ``refractory_seconds`` keeps one click from being counted twice.

    Returns a dictionary: ``times`` (seconds from the start of the window),
    ``amplitudes``, ``channels`` (the label that carried each click) and the
    threshold that was used.
    """

    if rate <= 0:
        raise ValueError("rate must be positive.")
    strength, carrier = onset_signal(eeg)
    median = float(np.median(strength))
    mad = float(np.median(np.abs(strength - median)))
    scale = mad if mad > 0 else (float(np.std(strength)) or 1.0)
    threshold = median + float(mad_multiple) * scale

    refractory = max(int(round(float(refractory_seconds) * rate)), 1)
    times: list[float] = []
    amplitudes: list[float] = []
    channels: list[str] = []
    last = -refractory - 1
    for index in np.flatnonzero(strength > threshold):
        if index - last < refractory:
            continue
        last = int(index)
        times.append(round(float(index) / rate, 9))
        amplitudes.append(float(strength[index]))
        if channel_names is not None:
            channels.append(str(channel_names[int(carrier[index])]))
    return {
        "times": tuple(times),
        "amplitudes": tuple(amplitudes),
        "channels": tuple(channels),
        "threshold": threshold,
        "median": median,
        "mad": mad,
        "mad_multiple": float(mad_multiple),
    }


def match_clicks(
    scheduled: tuple[float, ...],
    detected: tuple[float, ...],
    *,
    tolerance: float,
) -> dict:
    """Pair each scheduled click with the detected one nearest to it.

    Nearest-within-tolerance, one-to-one and in order: a detection may not be
    claimed by two scheduled clicks, and a scheduled click with no detection
    inside the tolerance is counted as *missed* rather than quietly dropped -
    because "we found 4 of 30 clicks" and "we found 30 of 30" are different
    claims about the same number.
    """

    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")
    used: set[int] = set()
    pairs: list[tuple[float, float]] = []
    missed: list[float] = []
    for moment in scheduled:
        best = None
        for index, seen in enumerate(detected):
            if index in used:
                continue
            if best is None or abs(seen - moment) < abs(detected[best] - moment):
                best = index
        if best is None or abs(detected[best] - moment) > tolerance:
            missed.append(float(moment))
            continue
        used.add(best)
        pairs.append((float(moment), float(detected[best])))
    return {
        "pairs": tuple(pairs),
        "missed": tuple(missed),
        "extra": tuple(
            float(value) for index, value in enumerate(detected) if index not in used
        ),
    }


def summarise(
    pairs: tuple[tuple[float, float], ...],
    *,
    mismatch: tuple[float, ...] = (),
    extra: tuple[float, ...] = (),
    tolerance: float = TOLERANCE_SECONDS,
    method: str = "cable",
    interval: float | None = None,
) -> dict:
    """The offset, its spread, and whether it is one to trust.

    The offset is the **median** of the per-click differences, not the mean: a
    single mis-detected click - a movement artifact in the same second - is
    exactly the kind of outlier a mean would carry into the answer. The
    uncertainty is the median absolute deviation of the same differences, so
    the two numbers describe the same population.
    """

    offsets = np.asarray([seen - moment for moment, seen in pairs], dtype=np.float64)
    record: dict = {
        "method": str(method),
        "tolerance_seconds": float(tolerance),
        "clicks_scheduled": int(len(pairs) + len(mismatch)),
        "clicks_matched": int(len(pairs)),
        "clicks_missed": int(len(mismatch)),
        "clicks_unmatched_detections": int(len(extra)),
        "sign_convention": (
            "offset = click time on the chain's axis minus the time playback was "
            "requested; positive means the sound reached the EEG later than the "
            "timeline says"
        ),
    }
    if offsets.size == 0:
        record.update(
            {
                "status": "not measurable",
                "offset_seconds": None,
                "uncertainty_seconds": None,
                "reason": (
                    "no click was matched inside the tolerance. Check the loopback "
                    "path, the output volume and the amplifier gain - and that the "
                    "player actually started."
                ),
            }
        )
        return record
    median = float(np.median(offsets))
    spread = float(np.median(np.abs(offsets - median)))
    zero_pairs = float(np.mean(np.abs(offsets) <= tolerance))
    record.update(
        {
            "status": "measured",
            "offset_seconds": round(median, 9),
            "uncertainty_seconds": round(spread, 9),
            "offset_min_seconds": round(float(offsets.min()), 9),
            "offset_max_seconds": round(float(offsets.max()), 9),
            "within_tolerance_fraction": round(zero_pairs, 6),
        }
    )
    if interval:
        detected = np.diff(np.asarray([seen for _, seen in pairs]))
        if detected.size:
            record["detected_interval_seconds"] = round(float(np.median(detected)), 9)
            record["interval_error_seconds"] = round(
                float(np.median(detected)) - float(interval), 9
            )
    if method == "acoustic":
        record["what_this_number_includes"] = (
            "the acoustic path from the headphones to the electrode AND the "
            "auditory evoked response. It is an upper bound on the electrical "
            "loopback offset, not the value to compensate the demo with."
        )
    else:
        record["what_this_number_includes"] = (
            "the sound card's output buffer, the cable, the amplifier's own "
            "conversion and the LSL delivery to this process - every term between "
            "'playback was requested' and 'the sample carries the click'."
        )
    if spread > tolerance:
        record["status"] = "measured, but the spread exceeds the tolerance"
        record["reason"] = (
            f"the per-click spread is {spread:.4f} s, above the +-{tolerance:g} s "
            "the plan allows. Re-run before trusting the alignment; if it repeats, "
            "the audio path's jitter is the limit, not the measurement."
        )
    return record


def write_profile(path: Path, record: dict, *, context: dict) -> dict:
    """Write the profile command 2 reads, and say what it was measured on."""

    profile = {
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "offset_seconds": record.get("offset_seconds"),
        "uncertainty_seconds": record.get("uncertainty_seconds"),
        "status": record.get("status"),
        "method": record.get("method"),
        "tolerance_seconds": float(record.get("tolerance_seconds", TOLERANCE_SECONDS)),
        "measurement": record,
        "context": context,
        "note": (
            "Read by scripts.auditory_ui.live --calibration. A profile whose status "
            "is not 'measured' must not be used to claim the system is aligned."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2, default=str), encoding="utf-8")
    return profile


def play_file(path: Path) -> dict:
    """Ask the system's own player to start the file; report how, not whether.

    No audio library is used on purpose: this repository has no audio dependency
    (``sounddevice``/``soundfile`` are not installed), and adding one to play a
    calibration file would change the runtime for every other path. The
    consequence is stated rather than hidden: the instant recorded here is when
    the *request* was made, and the player's own start-up delay is inside the
    measured offset, exactly as a browser's is.
    """

    if sys.platform == "win32":
        import os

        os.startfile(str(path))  # noqa: S606 - the operator's own file
        return {"how": "os.startfile (the system's default player)"}
    command = ["open", str(path)] if sys.platform == "darwin" else ["xdg-open", str(path)]
    subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return {"how": " ".join(command)}


def acquire(
    args, *, on_ready=None
) -> tuple[object, object, list, list, dict]:
    """Resolve the amplifier, connect, and read for ``--seconds``.

    Returns ``(source, stream, rows, times, facts)``. The first block is read
    before ``on_ready`` is called, because the chain's time axis does not exist
    until one sample has been placed - and the playback instant has to be
    expressed on that axis.
    """

    row = wait_for_outlet(
        name=args.stream_name,
        source_id=args.source_id,
        stream_type=args.stream_type,
        timeout=float(args.resolve_seconds),
        interval=1.0,
        expected_channels=None if args.stream_type else args.expected_channels,
    )
    print(f"resolved : name={row['name']!r} type={row['stype']!r} "
          f"channels={row['n_channels']} sfreq={row['sfreq']:g} Hz "
          f"source_id={row['source_id']!r}", flush=True)
    stream = open_inlet(row, bufsize=float(args.bufsize),
                        connect_timeout=float(args.connect_seconds))
    facts = channel_facts(stream)

    from nova2026.streaming.preflight import validate_source

    from scripts.getlive.ant_source import AntStreamSource

    validate_source(stream, sfreq=float(args.sfreq),
                    channels=tuple(args.electrodes),
                    source_unit_exponent=unit_exponent(args.source_units))
    source = AntStreamSource(
        stream,
        channels=tuple(args.electrodes),
        sfreq=float(args.sfreq),
        reference=args.reference,
        upstream_processing="calibration run: clicks through a loopback path",
        kind="loopback_calibration",
        source_unit_exponent=unit_exponent(args.source_units),
        block_samples=int(args.block_samples),
        timebase=args.timebase,
        pre_resample_sfreq=None,
        max_lag_seconds=3.0,
        no_data_timeout=float(args.no_data_timeout),
    )

    import threading

    stop = threading.Event()
    rows: list[np.ndarray] = []
    times: list[np.ndarray] = []
    totals = {"samples": 0}
    ready = {"at": None}
    deadline = time.monotonic() + float(args.seconds)

    def consume() -> None:
        for _end, data, placed in source.chunks(stop):
            rows.append(np.asarray(data, dtype=np.float64))
            times.append(np.asarray(placed, dtype=np.float64))
            totals["samples"] += int(len(placed))
            if ready["at"] is None:
                ready["at"] = (float(placed[-1]), time.perf_counter())
                if on_ready is not None:
                    on_ready(ready["at"])
            if time.monotonic() >= deadline:
                stop.set()

    thread = threading.Thread(target=consume, name="calibration-reader", daemon=True)
    thread.start()
    started = time.monotonic()
    while thread.is_alive() and time.monotonic() - started < float(args.seconds) + 15.0:
        time.sleep(0.1)
        if time.monotonic() >= deadline:
            stop.set()
    stop.set()
    thread.join(timeout=10.0)
    return source, stream, rows, times, facts


def live(args) -> int:
    """Acquire, play the click train, find the clicks and report the offset."""

    from mne_lsl.lsl import local_clock

    wave, scheduled = click_track(
        seconds=float(args.file_seconds),
        rate=int(args.audio_rate),
        interval=float(args.interval),
        lead=float(args.lead),
        click_ms=float(args.click_ms),
        amplitude=float(args.amplitude),
    )
    wav = Path(args.wav)
    wav_info = write_click_wav(wav, wave, int(args.audio_rate))
    print(f"click track: {wav_info['seconds']}s, {len(scheduled)} clicks, "
          f"{wav_info['sample_rate']} Hz -> {wav_info['path']}", flush=True)
    print(f"schedule   : click k at t = {args.lead:g} + k*{args.interval:g} s", flush=True)

    played: dict = {}

    def on_ready(anchor: tuple[float, float]) -> None:
        """Play the file as soon as the chain's clock exists.

        Two clocks are read back to back - the LSL stamp clock and this
        process's monotonic clock - so the offset between them is a single
        paired reading at the instant the request was made, and nothing here
        assumes they are the same clock.
        """

        before = float(local_clock())
        wall = time.perf_counter()
        info = play_file(wav)
        after = float(local_clock())
        played.update(
            {
                "eeg_time_at_request": before - float(anchor_origin(anchor)),
                "wall_seconds": wall,
                "lsl_before": before,
                "lsl_after": after,
                "paired_offset_seconds": (before + after) / 2.0 - wall,
                **info,
            }
        )
        print(f"playback    : requested at chain time "
              f"{played['eeg_time_at_request']:.6f} s via {info['how']}", flush=True)

    source = None
    stream = None
    anchor_values: list[float] = []

    def anchor_origin(anchor: tuple[float, float]) -> float:
        # The chain's axis starts at the first placed sample; `chunks` rebases
        # every stamp by it, and the diagnostics keep the raw first stamp.
        first = source.diagnostics.first_stamp if source is not None else None
        return float(first if first is not None else anchor[0])

    try:
        source, stream, rows, times, facts = acquire(args, on_ready=on_ready)
    except RuntimeError as error:
        print(f"\nREFUSED: {error}", file=sys.stderr, flush=True)
        return REFUSAL
    finally:
        if stream is not None and getattr(stream, "connected", False):
            stream.disconnect()

    if not rows:
        print("\nREFUSED: no samples arrived, so there is nothing to measure.",
              file=sys.stderr, flush=True)
        return REFUSAL
    eeg = np.concatenate(rows, axis=0)
    placed = np.concatenate(times, axis=0)
    print(f"acquired    : {eeg.shape[0]} samples over {placed[-1]:.2f} s "
          f"({eeg.shape[1]} channels, {args.sfreq:g} Hz)", flush=True)

    found = detect_click_times(
        eeg, float(args.sfreq), channel_names=tuple(args.electrodes),
        mad_multiple=float(args.mad_multiple),
        refractory_seconds=float(args.refractory),
    )
    print(f"detected    : {len(found['times'])} transient(s) above "
          f"{found['threshold']:.3g} uV/sample "
          f"(median {found['median']:.3g}, MAD {found['mad']:.3g})", flush=True)
    if found["channels"]:
        carriers = {name: found["channels"].count(name) for name in set(found["channels"])}
        print(f"carried by  : {carriers}", flush=True)

    if not played:
        print("\nREFUSED: playback was never requested (no first block arrived "
              "in time).", file=sys.stderr, flush=True)
        return REFUSAL
    request_time = float(played["eeg_time_at_request"])
    scheduled_on_eeg = tuple(request_time + moment for moment in scheduled)
    matched = match_clicks(
        scheduled_on_eeg, found["times"], tolerance=float(args.match_tolerance)
    )
    record = summarise(
        matched["pairs"], mismatch=matched["missed"], extra=matched["extra"],
        tolerance=TOLERANCE_SECONDS, method=args.method, interval=float(args.interval),
    )
    context = {
        "outlet": {"name": facts["name"], "type": facts["stype"],
                   "source_id": facts["source_id"], "sfreq": facts["sfreq"],
                   "n_channels": facts["n_channels"]},
        "electrodes": list(args.electrodes),
        "source_units": args.source_units,
        "timebase": args.timebase,
        "audio": {**wav_info, "interval_seconds": float(args.interval),
                  "lead_seconds": float(args.lead),
                  "request_at_chain_time_seconds": request_time,
                  "player": played.get("how")},
        "detected": {"count": len(found["times"]), "threshold": found["threshold"],
                     "channels": list(dict.fromkeys(found["channels"]))},
        "acquired_samples": int(eeg.shape[0]),
        "crosstalk_note": (
            "the click appears through the amplifier's own anti-alias filter, so "
            "its shape is not the played waveform; only its ONSET is used, and an "
            "onset is what the first-difference detector measures"
        ),
    }
    profile = write_profile(Path(args.profile), record, context=context)
    print(flush=True)
    for key in ("status", "offset_seconds", "uncertainty_seconds", "clicks_matched",
                "clicks_missed", "clicks_unmatched_detections",
                "detected_interval_seconds", "interval_error_seconds", "reason"):
        if key in record:
            print(f"{key:32s} {record[key]}", flush=True)
    print(f"\nprofile     : {args.profile}", flush=True)
    print(f"\nUse it:  python -B -m scripts.auditory_ui.live --calibration "
          f"{args.profile}", flush=True)
    if record["status"] != "measured":
        print("\nThe offset is NOT usable as measured. Do not claim the system is "
              "aligned.", file=sys.stderr, flush=True)
        return REFUSAL
    if abs(record["offset_seconds"]) > TOLERANCE_SECONDS:
        print(f"\nNote: the measured offset {record['offset_seconds']:+.4f} s exceeds "
              f"the +-{TOLERANCE_SECONDS:g} s budget. Command 2 applies the anchor it "
              "derives from the browser's own playback position; this number says how "
              "much delay sits outside that anchor, and it is what the residual "
              "alignment claim has to be made against.", flush=True)
    return 0


def make_clicks(args) -> int:
    """Write the click track without touching any hardware."""

    wave, times = click_track(
        seconds=float(args.file_seconds), rate=int(args.audio_rate),
        interval=float(args.interval), lead=float(args.lead),
        click_ms=float(args.click_ms), amplitude=float(args.amplitude),
    )
    info = write_click_wav(Path(args.wav), wave, int(args.audio_rate))
    print(f"wrote {info['path']}: {info['seconds']}s, {info['sample_rate']} Hz, "
          f"{len(times)} clicks, both ears identical", flush=True)
    for index, moment in enumerate(times[:5]):
        print(f"  click {index}: t = {moment:g} s", flush=True)
    if len(times) > 5:
        print(f"  ... click {len(times) - 1}: t = {times[-1]:g} s", flush=True)
    print("Play this file and run --live, or --analyse over a saved acquisition.",
          flush=True)
    return 0


def analyse(args) -> int:
    """Re-measure from a saved acquisition, so a number can be rechecked offline.

    The file is what ``--live`` would have collected: a ``.npz`` with ``eeg``
    (samples x channels), ``times`` (the chain's own axis) and, for a real
    measurement, ``request_time`` (the chain time at which playback was asked
    for).
    """

    path = Path(args.analyse)
    if not path.is_file():
        print(f"error: {path} not found", file=sys.stderr, flush=True)
        return REFUSAL
    saved = np.load(path, allow_pickle=False)
    eeg = np.asarray(saved["eeg"], dtype=np.float64)
    rate = float(saved["rate"]) if "rate" in saved else float(args.sfreq)
    names = tuple(str(name) for name in saved["channels"]) if "channels" in saved else None
    request_time = float(saved["request_time"]) if "request_time" in saved else 0.0
    _, scheduled = click_track(
        seconds=float(args.file_seconds), rate=int(args.audio_rate),
        interval=float(args.interval), lead=float(args.lead),
        click_ms=float(args.click_ms), amplitude=float(args.amplitude),
    )
    found = detect_click_times(eeg, rate, channel_names=names,
                               mad_multiple=float(args.mad_multiple),
                               refractory_seconds=float(args.refractory))
    matched = match_clicks(
        tuple(request_time + moment for moment in scheduled), found["times"],
        tolerance=float(args.match_tolerance),
    )
    record = summarise(matched["pairs"], mismatch=matched["missed"],
                       extra=matched["extra"], tolerance=TOLERANCE_SECONDS,
                       method=args.method, interval=float(args.interval))
    print(json.dumps(record, indent=2, default=str), flush=True)
    return 0 if record["status"] == "measured" else REFUSAL


def build_parser() -> argparse.ArgumentParser:
    """Build the calibration command line."""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_argument_group("what to do")
    mode.add_argument("--procedure", action="store_true",
                      help="print the operator procedure and exit")
    mode.add_argument("--make-clicks", action="store_true",
                      help="write the click track and print its schedule, no hardware")
    mode.add_argument("--live", action="store_true",
                      help="acquire from the amplifier and measure the offset")
    mode.add_argument("--analyse", default=None, metavar="NPZ",
                      help="re-measure from a saved acquisition")

    amp = parser.add_argument_group("amplifier")
    amp.add_argument("--stream-name", default=None)
    amp.add_argument("--source-id", default=None)
    amp.add_argument("--stream-type", default="EEG")
    amp.add_argument("--expected-channels", type=int, default=DEFAULT_EXPECTED_CHANNELS)
    amp.add_argument("--resolve-seconds", type=float, default=15.0)
    amp.add_argument("--connect-seconds", type=float, default=10.0)
    amp.add_argument("--bufsize", type=float, default=30.0)
    amp.add_argument("--block-samples", type=int, default=25)
    amp.add_argument("--sfreq", type=float, default=DEFAULT_SFREQ)
    amp.add_argument("--source-units", default="uV", choices=("V", "mV", "uV", "nV"))
    amp.add_argument("--timebase", default="grid", choices=("grid", "stamps"))
    amp.add_argument("--model", default=DEFAULT_MODEL)
    amp.add_argument("--electrodes", default=None,
                     help="comma-separated labels; default: the model's own list")
    amp.add_argument("--no-data-timeout", type=float, default=5.0)
    amp.add_argument("--reference", default="not asserted (calibration run)")

    track = parser.add_argument_group("click track")
    track.add_argument("--wav", default=DEFAULT_WAV)
    track.add_argument("--audio-rate", type=int, default=DEFAULT_AUDIO_RATE)
    track.add_argument("--file-seconds", type=float, default=30.0)
    track.add_argument("--seconds", type=float, default=30.0,
                       help="how long to acquire")
    track.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_SECONDS)
    track.add_argument("--lead", type=float, default=DEFAULT_LEAD_SECONDS,
                       help="silence before the first click, so playback has settled")
    track.add_argument("--click-ms", type=float, default=DEFAULT_CLICK_MS)
    track.add_argument("--amplitude", type=float, default=0.9)
    track.add_argument("--method", default="cable", choices=METHODS)

    detect = parser.add_argument_group("detection")
    detect.add_argument("--mad-multiple", type=float, default=12.0)
    detect.add_argument("--refractory", type=float, default=0.5)
    detect.add_argument("--match-tolerance", type=float, default=0.25,
                        help="how far a detected click may sit from its scheduled "
                             "instant and still be that click")

    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    return parser


def main(argv=None) -> int:
    """Dispatch the four things this command can do."""

    args = build_parser().parse_args(argv)
    if args.procedure:
        print(PROCEDURE, flush=True)
        return 0
    if args.make_clicks:
        return make_clicks(args)
    try:
        args.electrodes = tuple(
            part.strip() for part in (args.electrodes or "").split(",") if part.strip()
        ) or model_electrodes(args.model)
    except (ValueError, RuntimeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr, flush=True)
        return REFUSAL
    if math.isclose(float(args.seconds), float(args.file_seconds)) is False and (
        float(args.seconds) < float(args.lead) + float(args.interval)
    ):
        print("error: --seconds must cover at least the lead and one interval",
              file=sys.stderr, flush=True)
        return REFUSAL
    if args.analyse:
        return analyse(args)
    if args.live:
        return live(args)
    print(PROCEDURE, flush=True)
    print("Nothing was done: pass --make-clicks, --live or --analyse.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
