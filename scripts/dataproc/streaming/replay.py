"""Exercise the real two-thread runner with PlayerLSL, then compare chunk replay.

Run from the repository root with ``python -m scripts.dataproc.streaming.replay``.
All recordings are read-only; replay and verification arrays stay in memory.
"""

import argparse
import json
from pathlib import Path
from uuid import uuid4

import mne
import numpy as np
from mne_lsl.player import PlayerLSL

from .buffer_factory import create_buffer
from .config import CA208_CHANNELS, EEG_CHANNELS, StreamConfig
from .streamer import Streamer


def synthetic_recording(seconds: float = 90.0, sfreq: float = 500.0) -> mne.io.BaseRaw:
    """Create a known-amplitude, shuffled CA-208 recording in volts.

    Every channel has a distinct amplitude. EOG is deliberately large to verify
    that it is kept separate and cannot invalidate otherwise clean EEG windows.
    """

    names = list(reversed(CA208_CHANNELS))
    times = np.arange(round(seconds * sfreq)) / sfreq
    data = []
    for name in names:
        amplitude_uv = 1000.0 if name == "EOG" else 10.0 + CA208_CHANNELS.index(name)
        data.append(amplitude_uv * 1e-6 * np.sin(2 * np.pi * 10 * times))
    raw = mne.io.RawArray(
        np.asarray(data), mne.create_info(names, sfreq, "eeg"), verbose=False
    )
    raw.set_channel_types({"EOG": "eog"})
    return raw


def read_recording(path: Path) -> mne.io.BaseRaw:
    """Read an existing recording without modifying or re-exporting it."""

    if path.suffix.lower() == ".set":
        return mne.io.read_raw_eeglab(path, preload=False, verbose=False)
    if path.suffix.lower() == ".cnt":
        return mne.io.read_raw_ant(path, preload=False, verbose=False)
    return mne.io.read_raw(path, preload=False, verbose=False)


def run_replay(
    raw: mne.io.BaseRaw,
    duration: float = 15.0,
    source_unit_exponent: int = 0,
    synthetic: bool = False,
) -> dict:
    """Replay through LSL and check timing, counts, units, and offline equivalence.

    Args:
        raw: Read-only source recording. A private in-memory copy is used.
        duration: Wall-clock acquisition duration in seconds.
        source_unit_exponent: Player voltage multiplier; 0 is V and -6 is uV.
        synthetic: Whether to additionally verify the known sine amplitudes.

    Returns:
        JSON-compatible validation results. Failed comparisons raise assertions.

    Notes:
        Capturing raw chunks and output windows is a bounded test convenience,
        not part of the production runner. No recording or model is written.
    """

    if not np.isfinite(duration) or not 4 < duration <= 300:
        raise ValueError(
            "Replay duration must be greater than 4 and at most 300 seconds."
        )
    # Avoid silently looping across a recording boundary during validation.
    if raw.times[-1] < duration + 5:
        raise ValueError(
            "The recording must cover replay duration plus 5 seconds of setup."
        )
    recording = (
        raw.copy().crop(tmin=0, tmax=min(raw.times[-1], duration + 10)).load_data()
    )
    for onset, description in zip(
        recording.annotations.onset, recording.annotations.description
    ):
        internal = 0 < onset - recording.first_time < recording.times[-1]
        rejected = "boundary" in description.lower() or description.lower().startswith(
            "bad"
        )

        if internal and rejected:
            raise ValueError(
                "Replay requires a continuous segment without internal "
                "bad/boundary annotations."
            )
    available = dict(zip(recording.ch_names, recording.get_channel_types()))
    if not set(EEG_CHANNELS).issubset(available):
        raise ValueError("The recording is missing required shared EEG channels.")
    eog = ("EOG",) if available.get("EOG") == "eog" else ()
    source_id = f"nova-replay-{uuid4().hex}"
    config = StreamConfig(
        input_sfreq=float(recording.info["sfreq"]),
        source_unit_exponent=source_unit_exponent,
        input_reference="synthetic CPz contract"
        if synthetic
        else "recording-native; unconfirmed",
        upstream_processing="known synthetic sine"
        if synthetic
        else "recording-native; unconfirmed",
        stream_name="NOVA-Replay",
        source_id=source_id,
        eog_channels=eog,
    )
    # Keep the original file's channel order to exercise selection and reordering.
    player = PlayerLSL(
        recording,
        chunk_size=10,
        n_repeat=1,
        name=config.stream_name,
        source_id=source_id,
        annotations=False,
    )
    voltage_names = [
        name for name, kind in available.items() if kind in ("eeg", "eog", "ecg")
    ]
    player.set_channel_units({name: source_unit_exponent for name in voltage_names})
    chunks = []
    windows = []
    forwarded = []
    streamer = Streamer(config, pipeline=forwarded.append, on_window=windows.append)

    def capture(data, timestamps, info):
        """Retain only the samples actually received, in canonical queue order."""

        contract = streamer.connection.channel_selection
        assert contract is not None
        chunks.append((contract.apply_contract(data, axis=1), timestamps.copy()))

        return data, timestamps

    try:
        player.start()
        streamer.initialize()
        assert streamer.connection.stream is not None
        streamer.connection.stream.add_callback(capture)
        stats = streamer.stream(duration=duration)
    finally:
        streamer.close()

        if player.running:
            player.stop()
    assert streamer.acquisition_thread is not None
    assert streamer.connection.stream is not None
    assert not streamer.acquisition_thread.is_alive(), "Acquisition worker leaked."
    assert not streamer.connection.stream.connected, "LSL inlet leaked."
    assert stats.valid_windows > 0, "Replay produced no valid windows."
    assert all(window.valid for window in forwarded), (
        "Invalid window reached the pipeline."
    )

    # A last callback can race the duration boundary. Compare only consumed input.
    data = np.concatenate([chunk[0] for chunk in chunks])[: stats.input_samples]
    timestamps = np.concatenate([chunk[1] for chunk in chunks])[: stats.input_samples]
    if synthetic:
        # Independent scale/order oracle: every EEG column is the same sine,
        # multiplied by its known electrode-specific amplitude.
        amplitudes = np.array(
            [10 + CA208_CHANNELS.index(name) for name in config.eeg_channels]
        )
        volts = data[:, : len(amplitudes)] * 10.0**source_unit_exponent
        normalized = volts / (amplitudes[None, :] * 1e-6)
        np.testing.assert_allclose(
            normalized,
            np.repeat(normalized[:, :1], len(amplitudes), axis=1),
            atol=1e-10,
        )
        assert 0.99 < np.max(np.abs(normalized)) <= 1.000001

    offline = create_buffer(config, (source_unit_exponent,) * len(config.channels))
    expected = []
    # Deliberately use different chunk boundaries from the LSL player.
    for start in range(0, len(data), 137):
        offline.feed((data[start : start + 137], timestamps[start : start + 137]))
        expected.extend(offline.spit())
    assert offline.total_written == stats.output_samples
    expected_count = max(
        0, 1 + (stats.output_samples - offline.window_size) // offline.step_size
    )
    assert len(windows) == len(expected) == expected_count
    max_error = 0.0
    for actual, reference in zip(windows, expected):
        np.testing.assert_allclose(actual.data, reference.data, rtol=1e-9, atol=1e-8)
        np.testing.assert_allclose(actual.eog, reference.eog, rtol=1e-9, atol=1e-8)
        np.testing.assert_allclose(
            actual.timestamps, reference.timestamps, rtol=0, atol=1e-8
        )
        assert actual.valid == reference.valid and actual.reasons == reference.reasons
        assert actual.start_sample == reference.start_sample
        np.testing.assert_allclose(
            np.diff(actual.timestamps),
            1 / config.output_sfreq,
            rtol=0,
            atol=1e-8,
        )
        max_error = max(max_error, float(np.max(np.abs(actual.data - reference.data))))
    return {
        "source": "synthetic" if synthetic else "recording",
        "source_unit_exponent": source_unit_exponent,
        "eeg_channels": len(config.eeg_channels),
        "eog_channels": len(eog),
        "stats": stats.to_dict(),
        "expected_window_count": expected_count,
        "offline_max_error_uv": max_error,
        "clean_shutdown": True,
    }


def main() -> None:
    """Run a synthetic or local-recording replay and print its validation report."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--units", choices=("V", "uV"), default="V")
    args = parser.parse_args()
    if args.recording:
        raw = read_recording(args.recording)
    else:
        raw = synthetic_recording(args.duration + 15)
    report = run_replay(
        raw, args.duration, 0 if args.units == "V" else -6, args.recording is None
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
