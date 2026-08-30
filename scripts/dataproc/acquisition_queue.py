from queue import Empty, Queue
from threading import Lock

import numpy as np
from mne import Info


TIMESTAMP_EPSILON = 1e-6  # 1 microsecond


class StreamDiscontinuityError(RuntimeError):
    pass


class StreamDataValidityError(RuntimeError):
    pass


class StreamDataLagError(RuntimeError):
    pass


class StreamDataValueError(RuntimeError):
    pass


class AcquisitionQueue:
    def __init__(
        self,
        sfreq: float,
        missed_sample_tolerance: int,
        max_allowed_lag: float,
        expected_channel_count: int,
    ) -> None:

        if sfreq <= 0:
            raise ValueError("sfreq must be positive.")

        if missed_sample_tolerance < 0:
            raise ValueError("missed_sample_tolerance cannot be negative.")

        if max_allowed_lag <= 0:
            raise ValueError("max_allowed_lag must be positive.")

        if expected_channel_count <= 0:
            raise ValueError("expected_channel_count must be positive.")

        self.sfreq: float = sfreq
        self.missed_sample_tolerance: int = missed_sample_tolerance
        self.max_allowed_lag: float = max_allowed_lag
        self.expected_channel_count: int = expected_channel_count

        self.last_enqueued_time: float | None = None
        self.prev_time: float | None = None
        self.callback_error: Exception | None = None

        self.acq_queue: Queue[tuple[np.ndarray, np.ndarray]] = Queue()

        self.lock: Lock = Lock()  # Protects shared timestamp and error state

    def callback(
        self,
        data: np.ndarray,
        timestamps: np.ndarray,
        info: Info,
    ) -> tuple[np.ndarray, np.ndarray]:
        # data: (n_samples, n_channels)
        # timestamps: (n_times,)

        # TODO: check if with throws an error or correctly waits for the thread to finish & how self.lock works
        with self.lock:
            if self.callback_error is not None:
                return data, timestamps

        try:
            self._validate_chunks(data, timestamps)

            queue_item = (
                data.copy(),
                timestamps.copy(),
            )

            with self.lock:
                self.acq_queue.put_nowait(queue_item)
                self.last_enqueued_time = float(timestamps[-1])

        except Exception as error:
            self._store_callback_error(error)

        return data, timestamps

    def _store_callback_error(
        self,
        error: Exception,
    ) -> None:

        with self.lock:
            if self.callback_error is None:
                self.callback_error = error

    def _raise_callback_error(self) -> None:

        with self.lock:
            callback_error = self.callback_error

        if callback_error is not None:
            raise callback_error

    def _check_timestamp_continuity(
        self,
        timestamps: np.ndarray,
    ) -> None:

        if not np.all(np.isfinite(timestamps)):
            raise StreamDataValidityError("Timestamps contain NaN or infinity.")

        timestamp_intervals = np.diff(timestamps)

        if np.any(timestamp_intervals <= 0):
            raise StreamDiscontinuityError(
                "Timestamps within the chunk are not strictly increasing."
            )

        expected_interval = 1.0 / self.sfreq

        maximum_interval = (
            self.missed_sample_tolerance + 1
        ) * expected_interval + TIMESTAMP_EPSILON

        if np.any(timestamp_intervals > maximum_interval):
            raise StreamDiscontinuityError("Missing samples detected inside the chunk.")

        if self.prev_time is not None:
            boundary_interval = float(timestamps[0]) - self.prev_time

            if boundary_interval <= 0:
                raise StreamDiscontinuityError("Overlapping chunks detected.")

            if boundary_interval > maximum_interval:
                raise StreamDiscontinuityError(
                    "Missing samples detected between chunks."
                )

        self.prev_time = float(timestamps[-1])

    def _check_chunk_values(
        self,
        data: np.ndarray,
    ) -> None:

        if not np.all(np.isfinite(data)):
            raise StreamDataValueError("Raw EEG data contains NaN or infinity.")

    def _validate_chunks(
        self,
        data: np.ndarray,
        timestamps: np.ndarray,
    ) -> None:

        if data.ndim != 2:
            raise StreamDataValidityError(
                "Expected data dimensions to be "
                "(samples, channels). "
                f"Received shape {data.shape}."
            )

        if timestamps.ndim != 1:
            raise StreamDataValidityError(
                "Expected timestamp dimensions to be "
                "(samples,). "
                f"Received shape {timestamps.shape}."
            )

        if data.shape[1] != self.expected_channel_count:
            raise StreamDataValidityError(
                "Expected "
                f"{self.expected_channel_count} channels. "
                f"Received {data.shape[1]}."
            )

        if data.shape[0] != timestamps.shape[0]:
            raise StreamDataValidityError(
                f"Received {data.shape[0]} samples but "
                f"{timestamps.shape[0]} timestamps."
            )

        if data.shape[0] <= 0:
            raise StreamDataValidityError("The received chunk contained no samples.")

        self._check_chunk_values(data)
        self._check_timestamp_continuity(timestamps)

    def _validate_consumer_lag(
        self,
        timestamps: np.ndarray,
    ) -> None:

        with self.lock:
            last_enqueued_time = self.last_enqueued_time

        if last_enqueued_time is None:
            raise RuntimeError("No enqueued timestamp is available.")

        last_dequeued_time = float(timestamps[-1])

        lag = last_enqueued_time - last_dequeued_time

        if lag > self.max_allowed_lag:
            raise StreamDataLagError(
                f"The program is {lag:.3f} seconds behind acquisition."
            )

    def get(
        self,
        timeout: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray]:

        self._raise_callback_error()

        try:
            data, timestamps = self.acq_queue.get(timeout=timeout)

        except Empty:
            self._raise_callback_error()
            raise

        self._raise_callback_error()
        self._validate_consumer_lag(timestamps)

        return data, timestamps
