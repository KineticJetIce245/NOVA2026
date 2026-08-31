from queue import Empty, Queue
from threading import Lock

import numpy as np
from mne import Info


TIMESTAMP_EPSILON = 1e-6  # 1 microsecond


class StreamDiscontinuityError(RuntimeError):
    """Raised when incoming stream timestamps are discontinuous."""

    pass


class StreamDataValidityError(RuntimeError):
    """Raised when an incoming stream chunk has an invalid structure."""

    pass


class StreamDataLagError(RuntimeError):
    """Raised when processing falls too far behind data acquisition."""

    pass


class StreamDataValueError(RuntimeError):
    """Raised when incoming EEG data contains invalid numerical values."""

    pass


class AcquisitionQueue:
    """Transfer validated EEG chunks from acquisition to processing.

    Incoming chunks are received through :meth:`callback`, validated, copied,
    and stored in a thread-safe FIFO queue. The processing thread retrieves
    chunks through :meth:`get`. Errors detected on the acquisition thread are
    stored and subsequently raised on the processing thread.

    Args:
        sfreq (float): Expected sampling frequency of the stream in Hz.
        missed_sample_tolerance (int): Maximum number of missing samples
            permitted between consecutive received samples.
        max_allowed_lag (float): Maximum permitted delay in seconds between
            the most recently enqueued and dequeued samples.
        expected_channel_count (int): Expected number of channels in every
            incoming chunk.

    Attributes:
        sfreq (float): Expected sampling frequency in Hz.
        missed_sample_tolerance (int): Permitted number of missing samples.
        max_allowed_lag (float): Maximum permitted consumer lag in seconds.
        expected_channel_count (int): Expected number of stream channels.
        last_enqueued_time (float | None): Timestamp of the most recently
            enqueued sample.
        prev_time (float | None): Timestamp of the final sample in the
            previously validated chunk.
        callback_error (Exception | None): First error captured on the
            acquisition callback thread.
        acq_queue (Queue[tuple[np.ndarray, np.ndarray]]): FIFO queue containing
            copied data and timestamp arrays.
        lock (Lock): Lock protecting shared timestamp and error state.

    Raises:
        ValueError: If a configuration argument is outside its permitted
            range.
    """

    def __init__(
        self,
        sfreq: float,
        missed_sample_tolerance: int,
        max_allowed_lag: float,
        expected_channel_count: int,
        indices_contract: tuple[int, ...] | None,
    ) -> None:
        """Initialize the acquisition queue.

        Args:
            sfreq (float): Expected sampling frequency of the stream in Hz.
            missed_sample_tolerance (int): Maximum number of missing samples
                permitted between consecutive received samples.
            max_allowed_lag (float): Maximum permitted consumer lag in
                seconds.
            expected_channel_count (int): Expected number of channels in each
                incoming chunk.

        Raises:
            ValueError: If ``sfreq``, ``max_allowed_lag``, or
                ``expected_channel_count`` is not positive, or if
                ``missed_sample_tolerance`` is negative.
        """

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
        self.indices_contract = indices_contract
        self.lock: Lock = Lock()  # Protects shared timestamp and error state

    def callback(
        self,
        data: np.ndarray,
        timestamps: np.ndarray,
        _: Info,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate and enqueue a newly acquired EEG chunk.

        This method is intended to be registered as an MNE-LSL stream
        callback. Valid chunks are copied into the acquisition queue. The
        first validation error is stored so it can later be raised by the
        processing thread through :meth:`get`.

        Args:
            data (np.ndarray): EEG data with shape
                ``(n_samples, n_channels)``.
            timestamps (np.ndarray): Sample timestamps with shape
                ``(n_samples,)``.
            info (Info): MNE metadata describing the connected stream.

        Returns:
            tuple[np.ndarray, np.ndarray]: The original data and timestamp
                arrays required by the MNE-LSL callback interface.
        """
        # data: (n_samples, n_channels)
        # timestamps: (n_times,)

        # TODO: check if with throws an error or correctly waits for the thread to finish & how self.lock works
        with self.lock:
            if self.callback_error is not None:
                return data, timestamps

        try:
            self._validate_chunks(data, timestamps)

            if self.indices_contract is None:
                queued_data = data.copy()
            else:
                queued_data = np.take(data, indices=self.indices_contract, axis=1)

            queue_item = (
                queued_data,
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
        """Store the first error raised on the acquisition thread.

        Subsequent callback errors do not replace the first stored error.

        Args:
            error (Exception): Error raised while validating or enqueueing an
                acquisition chunk.
        """

        with self.lock:
            if self.callback_error is None:
                self.callback_error = error

    def _raise_callback_error(self) -> None:
        """Raise an error previously stored by the acquisition callback.

        Raises:
            Exception: The first error captured on the acquisition thread.
        """

        with self.lock:
            callback_error = self.callback_error

        if callback_error is not None:
            raise callback_error

    def _check_timestamp_continuity(
        self,
        timestamps: np.ndarray,
    ) -> None:
        """Validate timestamp values and continuity.

        Timestamps must be finite and strictly increasing. Consecutive
        timestamps must not exceed the interval permitted by the configured
        sampling frequency and missing-sample tolerance. The first timestamp
        of a chunk must also follow the final timestamp of the previous chunk.

        Args:
            timestamps (np.ndarray): Sample timestamps with shape
                ``(n_samples,)``.

        Raises:
            StreamDataValidityError: If a timestamp is NaN or infinite.
            StreamDiscontinuityError: If timestamps are not strictly
                increasing, chunks overlap, or too many samples are missing.
        """

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
        """Validate the numerical values in an EEG chunk.

        Args:
            data (np.ndarray): EEG data with shape
                ``(n_samples, n_channels)``.

        Raises:
            StreamDataValueError: If the data contains NaN or infinity.
        """

        if not np.all(np.isfinite(data)):
            raise StreamDataValueError("Raw EEG data contains NaN or infinity.")

    def _validate_chunks(
        self,
        data: np.ndarray,
        timestamps: np.ndarray,
    ) -> None:
        """Validate the structure, values, and timing of an EEG chunk.

        Args:
            data (np.ndarray): EEG data expected to have shape
                ``(n_samples, n_channels)``.
            timestamps (np.ndarray): Timestamps expected to have shape
                ``(n_samples,)``.

        Raises:
            StreamDataValidityError: If the arrays have invalid dimensions,
                contain different sample counts, contain no samples, or the
                channel count is unexpected.
            StreamDataValueError: If the EEG data contains NaN or infinity.
            StreamDiscontinuityError: If the timestamps are discontinuous.
        """

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
        """Validate the delay between acquisition and processing.

        Consumer lag is measured as the difference between the final timestamp
        of the most recently enqueued chunk and the final timestamp of the
        chunk currently being dequeued.

        Args:
            timestamps (np.ndarray): Timestamps belonging to the dequeued
                chunk.

        Raises:
            RuntimeError: If no enqueued timestamp is available.
            StreamDataLagError: If consumer lag exceeds
                ``max_allowed_lag``.
        """

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
        """Remove and return the oldest validated acquisition chunk.

        The method checks for errors captured by the acquisition callback
        before and after waiting for a queue item. If no item arrives before
        the timeout, any newly stored callback error is raised before the
        queue timeout is propagated.

        Args:
            timeout (float, optional): Maximum number of seconds to wait for
                a chunk. Defaults to ``0.1``.

        Returns:
            tuple[np.ndarray, np.ndarray]: The oldest queued data and timestamp
                arrays.

        Raises:
            Empty: If no chunk becomes available before ``timeout`` and no
                callback error was stored.
            Exception: If the acquisition callback stored an error.
            StreamDataLagError: If processing is too far behind acquisition.
        """

        self._raise_callback_error()

        try:
            data, timestamps = self.acq_queue.get(timeout=timeout)

        except Empty:
            self._raise_callback_error()
            raise

        self._raise_callback_error()
        self._validate_consumer_lag(timestamps)

        return data, timestamps
