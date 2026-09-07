"""Simple two-thread acquisition and processing runner."""

import math
from collections.abc import Callable
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from time import monotonic

import numpy as np
from mne_lsl.lsl import local_clock

from .acquisition_queue import AcquisitionQueue, StreamDataLagError
from .circular_buffer import CircularBuffer
from .config import StreamConfig
from .processor import RunProcessor
from .recording import RunRecorder
from .run import RunSpec
from .spatial_operator import SpatialOperator
from .stats import StreamStats
from .streams import StreamConnection
from .window import EEGWindow


class Streamer:
    """Acquire on one worker thread and process on the calling thread.

    Args:
        config: Source and preprocessing settings for the run.
        pipeline: Optional callable receiving valid EEGWindow objects only.
            Pass an existing pipeline's rundown method if its tubes accept
            EEGWindow objects. The existing offline pipelines are not modified.
        on_window: Optional diagnostic observer receiving every window, including
            rejected windows. This is not the inference entry point.
        run: Identity, role, and explicitly selected artifact/baseline/model files.
        record_root: Optional parent directory for subject/session/run recordings.
        on_result: Receives the pipeline return value unchanged. For Pipeline.rundown,
            this is (result, final_data). Store or forward predictions here.
        pipeline_on_invalid: Opt in only when the pipeline explicitly handles rejected
            windows without inference, as RiemannPipeline does. Defaults to False.

    Notes:
        Call initialize(), then stream(). Call stop() to request shutdown from
        another thread or a callback. Every new run requires initialize() again.
        Pipeline and observer callbacks execute synchronously on the processing
        thread and must return promptly; Python cannot interrupt a stuck callback.
    """

    def __init__(
        self,
        config: StreamConfig,
        pipeline: Callable[[EEGWindow], object] | None = None,
        on_window: Callable[[EEGWindow], object] | None = None,
        run: RunSpec | None = None,
        record_root: Path | None = None,
        on_result: Callable[[object], object] | None = None,
        pipeline_on_invalid: bool = False,
    ) -> None:
        """Store callbacks and optional recording selections without starting IO."""

        if record_root is not None and run is None:
            raise ValueError("Recording requires an explicit RunSpec.")

        self.config = config.updated()
        self.run = run
        self.record_root = record_root
        self.recorder: RunRecorder | None = None
        self.processor: RunProcessor | None = None
        self.markers: Queue[tuple[float, str]] = Queue(maxsize=256)
        self.pipeline = pipeline
        self.on_result = on_result
        self.pipeline_on_invalid = pipeline_on_invalid
        self.on_window = on_window
        self.connection = StreamConnection(self.config)
        self.acquisition_queue: AcquisitionQueue | None = None
        self.circular_buffer: CircularBuffer | None = None
        self.acquisition_thread: Thread | None = None
        self.stop_event = Event()
        self.initialized = False
        self.running = False
        self.stats = StreamStats()

    def initialize(self) -> None:
        """Connect and build fresh processing state without starting a worker.

        Raises:
            RuntimeError: If a run is active or source metadata is incompatible.
        """

        worker_alive = (
            self.acquisition_thread is not None and self.acquisition_thread.is_alive()
        )

        if self.running or worker_alive:
            raise RuntimeError("Stop the current run before initializing again.")
        self.close()
        self.stop_event.clear()
        self.stats = StreamStats()
        self.markers = Queue(maxsize=256)
        self.recorder = None

        try:
            source = self.connection.connect()
            config = self.config
            contract = self.connection.channel_selection
            if contract is None:
                raise RuntimeError(
                    "Connection setup did not produce a channel contract."
                )
            self.acquisition_queue = AcquisitionQueue(
                config.input_sfreq,
                0,
                config.max_lag_seconds,
                len(config.channels),
                contract,
                max_chunks=config.queue_size,
                timestamp_tolerance=config.timestamp_tolerance,
                validate_signal=False,
            )
            operator = None
            if self.run is not None and self.run.artifact_path is not None:
                operator = SpatialOperator.load(self.run.artifact_path)

            self.processor = RunProcessor(config, operator)
            self.circular_buffer = self.processor.buffer

            if self.run is not None and self.record_root is not None:
                self.recorder = RunRecorder(self.record_root, self.run, config)

            source.add_callback(self.acquisition_queue.callback)
            self.initialized = True
        except BaseException:
            self.close()
            raise

    def _acquire(self) -> None:
        """Pull nonblocking LSL chunks; callbacks copy them to the bounded queue."""

        source = self.connection.stream
        queue = self.acquisition_queue

        if source is None or queue is None:
            raise RuntimeError("Initialize acquisition before starting the worker.")

        try:
            while not self.stop_event.is_set():
                source.acquire()
                if not source.connected:
                    # MNE-LSL may log an acquisition error and disconnect rather
                    # than raise it. Never let that look like a healthy source.
                    raise RuntimeError("The source disconnected during acquisition.")
                queue.raise_error()
                if source.n_new_samples:
                    # Callback data is already queued. Clear MNE's unread counter
                    # with a one-sample view rather than copying its entire buffer.
                    source.get_data(winsize=1 / self.config.input_sfreq, exclude=())
                self.stop_event.wait(self.config.poll_interval)
        except Exception as error:  # noqa: BLE001 - re-raised on the processing thread
            queue.store_error(error)

    def stream(self, duration: float | None = None) -> StreamStats:
        """Process windows until stopped, timed out, or interrupted by an error.

        Args:
            duration: Optional wall-clock duration in seconds, useful for replay.

        Returns:
            Counts and maximum observed lag for the completed run.

        Raises:
            TimeoutError: If no data arrives within the configured interval.
            Exception: Original acquisition, processing, or callback error.
        """

        if not self.initialized or self.running:
            raise RuntimeError("Call initialize() before each stream() run.")

        if duration is not None and (not math.isfinite(duration) or duration <= 0):
            raise ValueError("duration must be finite and positive.")
        queue = self.acquisition_queue
        circular_buffer = self.circular_buffer
        processor = self.processor

        if queue is None or circular_buffer is None or processor is None:
            raise RuntimeError("Processing was not initialized.")

        failure: BaseException | None = None
        self.running = True
        started = monotonic()
        last_data = started
        self.acquisition_thread = Thread(
            target=self._acquire, name="nova-eeg-acquisition"
        )
        try:
            if self.recorder is not None:
                source = self.connection.stream
                contract = self.connection.channel_selection
                if source is None or contract is None:
                    raise RuntimeError("Source metadata is unavailable.")
                self.recorder.open(
                    {
                        "inlet_channels": contract.eeg_channel_names,
                        "selected_inlet_channels": contract.selected_channel_names,
                        "canonical_channels": self.config.channels,
                        "name": source.name,
                        "source_id": source.source_id,
                    },
                    track_windows=True,
                )
                if processor.operator is not None:
                    snapshot = self.recorder.metadata["inputs"]["artifact_path"][
                        "snapshot"
                    ]
                    saved = SpatialOperator.load(self.recorder.directory / snapshot)
                    if saved.artifact_id != processor.operator.artifact_id:
                        raise RuntimeError(
                            "The selected artifact changed during setup."
                        )

                self.recorder.event(local_clock(), "run_start", {})

            self.acquisition_thread.start()
            while not self.stop_event.is_set():
                queue.raise_error()
                self._write_markers()
                if duration is not None and monotonic() - started >= duration:
                    break
                try:
                    item = queue.get(timeout=0.05)
                except Empty:
                    if monotonic() - last_data >= self.config.no_data_timeout:
                        raise TimeoutError(
                            "No EEG data arrived before the no-data timeout."
                        )
                    continue
                last_data = monotonic()
                if self.recorder is not None:
                    self.recorder.write_chunk(item[0], item[1], processor.segment)

                if np.isfinite(item[1][-1]):
                    age = local_clock() - float(item[1][-1])
                    self.stats.max_queue_lag = max(self.stats.max_queue_lag, age)
                    if age > self.config.max_lag_seconds:
                        raise StreamDataLagError(
                            f"Input samples are {age:.3f} seconds old."
                        )
                    if age < -self.config.max_future_seconds:
                        raise StreamDataLagError(
                            "Input timestamps are ahead of the local LSL clock."
                        )

                windows = processor.feed(item)
                self.stats.chunks += 1
                self.stats.input_samples += len(item[0])
                if self.recorder is not None:
                    for event in processor.events:
                        timestamp = event["timestamp"]
                        self.recorder.event(
                            local_clock() if timestamp is None else timestamp,
                            event.get("kind", "recovery"),
                            event,
                        )

                for window in windows:
                    if self.stop_event.is_set():
                        break
                    queue.raise_error()
                    self._process_window(window)
        except BaseException as error:
            failure = error
            raise
        finally:
            self.stop_event.set()
            if self.acquisition_thread.ident is not None:
                self.acquisition_thread.join(timeout=2.0)
            self.running = False
            self.initialized = False
            if self.acquisition_thread.is_alive():
                if self.recorder is not None:
                    self.recorder.close(
                        "failed", self.stats.to_dict(), "Acquisition did not stop."
                    )
                raise RuntimeError(
                    "Acquisition did not stop; the inlet is still owned by its worker."
                )
            self.stats.output_samples = processor.output_samples
            self.stats.recoveries = processor.recoveries
            self.stats.interpolated_samples = processor.interpolated_samples
            self.stats.pending_interpolation_samples = len(
                processor.interpolator.pending
            )
            self.stats.discarded_chunks = processor.discarded_chunks
            self.stats.max_queue_size = queue.max_observed_size

            try:
                self.connection.disconnect()
                if self.recorder is not None and self.recorder.connection is not None:
                    for data, timestamps in queue.drain():
                        self.recorder.write_chunk(
                            data, timestamps, processor.segment, "not_processed"
                        )
                    self._write_markers()
                    error = failure if failure is not None else queue.callback_error
                    self.recorder.close(
                        "completed" if error is None else "failed",
                        self.stats.to_dict(),
                        None if error is None else repr(error),
                    )
            finally:
                if self.recorder is not None:
                    self.recorder.close(
                        "failed", self.stats.to_dict(), "Finalization failed."
                    )
        queue.raise_error()

        return self.stats

    def _process_window(self, window: EEGWindow) -> None:
        """Record validity and send only valid, timely windows to the pipeline."""

        lag = local_clock() - float(window.timestamps[-1])
        self.stats.max_window_lag = max(self.stats.max_window_lag, lag)

        if lag > self.config.max_lag_seconds:
            raise StreamDataLagError(f"Output window is {lag:.3f} seconds old.")
        self.stats.windows += 1

        if window.valid:
            self.stats.valid_windows += 1
        else:
            self.stats.invalid_windows += 1

        if self.pipeline is not None and (window.valid or self.pipeline_on_invalid):
            result = self.pipeline(window)
            if self.on_result is not None:
                self.on_result(result)

        if self.on_window is not None:
            self.on_window(window)

        if self.recorder is not None:
            self.recorder.event(
                float(window.timestamps[-1]),
                "window",
                {
                    "segment": window.segment,
                    "start_sample": window.start_sample,
                    "interpolated_samples": window.interpolated_samples,
                    "interpolated": window.interpolated,
                },
            )

    def stop(self) -> None:
        """Request shutdown; the processing owner joins the worker and closes LSL."""

        self.stop_event.set()

    def close(self) -> None:
        """Release a prepared but inactive source, including after setup failure."""

        worker_alive = (
            self.acquisition_thread is not None and self.acquisition_thread.is_alive()
        )

        if self.running or worker_alive:
            raise RuntimeError("Use stop() while a run is active.")
        self.connection.disconnect()
        self.initialized = False

    def mark(self, label: str, timestamp: float | None = None) -> None:
        """Queue a task marker in local LSL time from the task/controller thread.

        Args:
            label: Task label, e.g. blink, eyes_left, stimulus, or response.
            timestamp: Original event time; defaults to local_clock().

        Raises:
            RuntimeError: If no recording is active.
            queue.Full: If the bounded marker queue is not being consumed.
        """

        if not self.running or self.recorder is None:
            raise RuntimeError("Task markers require an active recorded run.")
        timestamp = local_clock() if timestamp is None else timestamp

        if not label.strip() or not math.isfinite(timestamp):
            raise ValueError("Markers need a label and a finite timestamp.")
        self.markers.put_nowait((timestamp, label))

    def _write_markers(self) -> None:
        """Drain task markers on the processing thread, the recorder's owner."""

        if self.recorder is None:
            return
        while True:
            try:
                timestamp, label = self.markers.get_nowait()
            except Empty:
                return
            self.recorder.event(timestamp, "marker", {"label": label})
