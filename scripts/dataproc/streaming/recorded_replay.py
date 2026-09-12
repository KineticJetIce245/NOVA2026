"""Reconstruct runs and calibrate from their recorded task markers."""

import argparse
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from .artifact_calibration import ArtifactCalibration
from .processor import RunProcessor
from .run_reader import RunReader
from .spatial_operator import SpatialOperator
from .window import EEGWindow


def replay_run(
    directory: Path,
    operator_path: Path | None = None,
    apply_saved_operator: bool = True,
) -> Iterator[EEGWindow]:
    """Reconstruct windows using original chunks and the same live processor.

    Args:
        directory: Completed run directory containing run.sqlite.
        operator_path: Explicit replacement operator, validated against the run.
        apply_saved_operator: Use the run's immutable operator snapshot by default.

    Yields:
        Windows with the same segment, source timing, validity, and correction.
    """

    reader = RunReader(directory)
    try:
        if reader.metadata["status"] != "completed":
            raise ValueError("Only completed runs can be replayed as complete data.")

        if operator_path is None and apply_saved_operator:
            operator_path = reader.input_path("artifact_path")
        operator = (
            None if operator_path is None else SpatialOperator.load(operator_path)
        )
        processor = RunProcessor(reader.config, operator)

        delivered = None

        if reader.metadata.get("track_windows", False):
            delivered = {
                (details["segment"], details["start_sample"])
                for _, kind, details in reader.events()
                if kind == "window"
            }

        for data, timestamps, disposition in reader.chunks():
            if disposition != "not_processed":
                for window in processor.feed((data, timestamps)):
                    if (
                        delivered is None
                        or (window.segment, window.start_sample) in delivered
                    ):
                        yield window
    finally:
        reader.close()


def calibration_epochs(directory: Path, event_seconds: float = 1.0) -> tuple:
    """Extract non-overlapping valid epochs around blink/eye-movement markers.

    Only complete epochs within a single valid window are used. Missing markers,
    overlaps, gaps, warm-up, and rejected epochs cause a clear error instead of
    silently changing the calibration sample set.
    """

    if not np.isfinite(event_seconds) or event_seconds <= 0:
        raise ValueError("Event duration must be finite and positive.")

    reader = RunReader(directory)
    try:
        if reader.metadata["run"]["role"] != "artifact_calibration":
            raise ValueError("Select a run recorded with role artifact_calibration.")
        config = reader.config
        identity = reader.metadata["run"]
        labels = {"blink", "eyes_left", "eyes_right", "eyes_up", "eyes_down"}
        times = sorted(
            time
            for time, kind, details in reader.events()
            if kind == "marker" and details["label"] in labels
        )
    finally:
        reader.close()

    if len(times) < 6 or np.any(np.diff(times) < event_seconds):
        raise ValueError("Record at least six separated blink/eye-movement markers.")
    sample_count = config.sample_count(event_seconds)
    if sample_count < 1 or event_seconds > config.window_seconds:
        raise ValueError("An epoch must fit inside a processing window.")

    epochs: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for window in replay_run(directory, apply_saved_operator=False):
        if not window.valid:
            continue

        for index, timestamp in enumerate(times):
            left = timestamp - event_seconds / 2
            if index in epochs or left < window.timestamps[0]:
                continue
            start = int(np.searchsorted(window.timestamps, left))
            stop = start + sample_count
            if stop <= len(window.data):
                epochs[index] = (
                    window.data[start:stop].copy(),
                    window.eog[start:stop].copy(),
                )

    if len(epochs) != len(times):
        missing = [index + 1 for index in range(len(times)) if index not in epochs]
        raise ValueError(
            f"Some calibration events overlap warm-up, gaps, or bad data: {missing}."
        )
    eeg = np.stack([epochs[index][0] for index in range(len(times))])
    eog = np.stack([epochs[index][1] for index in range(len(times))])
    return config, eeg, eog, identity


def calibrate_run(
    directory: Path, output: Path, n_components: int = 1
) -> SpatialOperator:
    """Fit, save, and plot a reproducible operator from a completed calibration run.

    The PNG shows held-out before/after EEG, measured EOG, and spatial weights.
    It is a review aid, not evidence that all neural signals are preserved.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    if output.suffix.lower() != ".npz":
        raise ValueError("Use an .npz filename for the saved operator.")

    plot_path = output.with_suffix(".png")
    report_path = output.with_suffix(".json")
    if any(path.exists() for path in (output, plot_path, report_path)):
        raise FileExistsError("Choose unused operator/report filenames.")

    config, eeg, eog, identity = calibration_epochs(directory)
    operator = ArtifactCalibration(config, n_components).fit(eeg, eog)
    operator.report["source_run"] = identity
    operator.report["source_sha256"] = hashlib.sha256(
        (directory / "run.sqlite").read_bytes()
    ).hexdigest()

    example = eeg[-1]
    corrected = operator.apply(example)
    channel = int(
        np.argmax(np.sum(np.asarray(operator.report["directions"]) ** 2, axis=1))
    )
    times = np.arange(len(example)) / config.output_sfreq
    figure, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
    axes[0].plot(times, example[:, channel], label="Before")
    axes[0].plot(times, corrected[:, channel], label="After")
    axes[0].set_title(f"Held-out event: {config.eeg_channels[channel]} (uV)")
    axes[0].legend()
    axes[1].plot(times, eog[-1])
    axes[1].set_title("Measured EOG (uV)")
    axes[1].set_xlabel("Seconds within held-out event")
    axes[2].plot(operator.report["directions"])
    axes[2].set_xticks(
        range(len(config.eeg_channels)), config.eeg_channels, rotation=90
    )
    axes[2].set_title("Removed spatial directions in saved EEG order")
    try:
        with plot_path.open("xb") as stream:
            figure.savefig(stream, format="png", dpi=130)
    finally:
        plt.close(figure)

    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(operator.report, stream, indent=2, allow_nan=False)
    operator.save(output)
    return operator


def main() -> None:
    """Calibrate an existing run or summarize its reconstructed windows."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--calibrate", type=Path, metavar="OPERATOR_NPZ")
    parser.add_argument("--components", type=int, default=1)
    args = parser.parse_args()

    if args.calibrate:
        operator = calibrate_run(args.directory, args.calibrate, args.components)
        print(json.dumps(operator.report, indent=2))
    else:
        valid = 0
        total = 0

        for window in replay_run(args.directory):
            total += 1
            valid += int(window.valid)
        print(json.dumps({"windows": total, "valid": valid, "rejected": total - valid}))


if __name__ == "__main__":
    main()
