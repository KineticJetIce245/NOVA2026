"""Connect the existing EEG processor to timestamped audio features."""

import numpy as np

from nova2026.auditory.alignment import EnvelopeBuffer
from nova2026.auditory.envelopes import EnvelopeExtractor
from scripts.dataproc.streaming.config import StreamConfig
from scripts.dataproc.streaming.processor import RunProcessor


class ReplayFailure(RuntimeError):
    """A stopped EEG run with its failure time on the replay clock."""

    def __init__(self, timestamp, message):
        super().__init__(message)
        self.timestamp = timestamp


def stream_config(trial, config, history=5.0, step=1.0):
    """Interchange EEG is in uV; input reference is declared, never guessed."""
    return StreamConfig(
        input_sfreq=round(trial.sample_rate, 6),
        source_unit_exponent=-6,
        input_reference=trial.reference,
        upstream_processing=trial.upstream_processing,
        stream_name="auditory-replay",
        eeg_channels=trial.channel_names,
        eog_channels=(),
        output_sfreq=config.sample_rate,
        bandpass=config.band,
        notch_frequency=None,
        window_seconds=history,
        step_seconds=step,
        buffer_seconds=history + 4,
        warmup_seconds=2.0,
    )


def replay_windows(
    trial,
    config,
    history=5.0,
    step=1.0,
    chunk_seconds=0.032,
    max_wait=3.0,
    audio_offset=0.0,
):
    """Yield aligned windows on a virtual clock; never precompute future audio.

    Audio sample zero is at trial start plus audio_offset. Processing outputs
    retain nominal source times and a separate conservative availability time.
    Pending windows wait a bounded time for the audio resampler, then reject.
    """
    processing = stream_config(trial, config, history, step)
    processor = RunProcessor(processing)
    extractor = EnvelopeExtractor(trial.audio_rate, config)
    envelopes = EnvelopeBuffer(config.sample_rate, history + max_wait + 5)
    audio_start = trial.timestamps[0] + audio_offset
    eeg_position = 0
    audio_position = 0
    envelope_position = 0
    pending = []
    now = float(trial.timestamps[0])
    end = min(
        float(trial.timestamps[-1]), audio_start + len(trial.audio) / trial.audio_rate
    )
    while now < end:
        now = min(now + chunk_seconds, end)
        audio_end = min(
            len(trial.audio), max(0, int((now - audio_start) * trial.audio_rate))
        )
        if audio_end > audio_position:
            values = extractor.feed(trial.audio[audio_position:audio_end])
            times = (
                audio_start
                + (envelope_position + np.arange(len(values))) / config.sample_rate
            )
            envelopes.feed(values, times, now)
            envelope_position += len(values)
            audio_position = audio_end
        eeg_end = np.searchsorted(trial.timestamps, now, side="right")
        if eeg_end > eeg_position:
            samples = trial.eeg[eeg_position:eeg_end]
            times = trial.timestamps[eeg_position:eeg_end]
            try:
                windows = processor.feed((samples, times))
            except RuntimeError as error:
                raise ReplayFailure(now, str(error)) from error
            for window in windows:
                pending.append((window, now))
            eeg_position = eeg_end
        remaining = []
        for window, created in pending:
            aligned = envelopes.align(window)
            if "audio_unavailable" in aligned.reasons and now - created < max_wait:
                remaining.append((window, created))
                continue
            aligned.available_at = max(now, aligned.available_at)
            yield aligned
        pending = remaining
    for window, created in pending:
        aligned = envelopes.align(window)
        aligned.valid = False
        aligned.reasons = tuple(set(aligned.reasons) | {"unprocessed_tail"})
        aligned.available_at = end
        yield aligned


def labels_for_window(trial, window):
    indices = np.searchsorted(trial.timestamps, window.timestamps, side="right") - 1
    indices = np.clip(indices, 0, len(trial.labels) - 1)
    return trial.labels[indices]


class LiveAuditoryAdapter:
    """Connect Streamer callbacks to prepared audio history and a result handoff.

    EnvelopeBuffer is owned by the processing thread. The caller must drain a
    bounded timestamped audio-feature queue on that thread before this callback.
    The audio controller owns the independent decision-expiry clock.
    """

    def __init__(self, pipeline, envelopes, handoff):
        self.pipeline = pipeline
        self.envelopes = envelopes
        self.handoff = handoff

    def rundown(self, window):
        aligned = self.envelopes.align(window)
        result = self.pipeline.rundown(aligned)
        self.handoff.put(result[0])
        return result
