"""Ridge fits from accumulated moments, arithmetically equal to ``RidgeDecoder``.

:meth:`~nova2026.auditory.decoder.RidgeDecoder.fit` is the reference: it walks
the training windows, builds each one's lag-stacked design matrix and adds
``design.T @ design`` into a covariance. That is the right arithmetic and the
wrong cost here: 320 trials over twenty folds would rebuild almost the same
covariance every time, at roughly 44 TFLOP per pass.

The identity used instead is that the design matrix is a lag stack of one
signal, so its Gram is block-Toeplitz in the lag index::

    X[t, (l, c)] = z[t + l, c]
    X.T @ X[(l1, c1), (l2, c2)] = sum_t z[t + l1, c1] * z[t + l2, c2]

That sum depends on ``l1`` and ``l2`` only through their difference, apart from
``lag`` boundary terms at each end of the trial. One pass of ``2 * lag + 1``
shifted products per trial therefore yields the whole Gram, and the boundary
terms are a handful of small products.

A covariance is additive over trials, so per-trial moments sum into any fold,
and a fold's normalization is applied afterwards -- exactly, not approximately.
The fold's mean and scale are known before the sum is assembled, and the
correction they imply is a rank-structured update of the raw moments.

What comes out is an ordinary :class:`RidgeDecoder` with ``weights``, ``mean``,
``scale`` and ``contract``, so nothing on the runtime path needs to know this
module exists. ``tests/test_aad_decoder.py`` asserts that a fit here and a
:meth:`RidgeDecoder.fit` fit agree to floating-point tolerance on the same data.
"""

import numpy as np

from nova2026.auditory.config import AuditoryConfig
from nova2026.auditory.decoder import RidgeDecoder


def design_rows(starts, length, lag):
    """The rows each window predicts, in the chain's own order.

    A window predicts every sample it holds except the last ``lag``: those rows
    would need EEG beyond the window, which is why ``RidgeDecoder.design`` drops
    them and why the row set is not quite the union of the window spans.
    """

    return [(int(start), int(start) + length - lag) for start in starts]


def _cumulative(values):
    return np.vstack([np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)])


def _interval_sums(cumulative, intervals):
    if not len(intervals):
        return np.zeros(cumulative.shape[1])
    edges = np.asarray(intervals, dtype=np.int64)
    return (cumulative[edges[:, 1]] - cumulative[edges[:, 0]]).sum(axis=0)


def _shifted_sums(cumulative, intervals, lag):
    """``sum over rows of z[t + l]`` for every lag, one row per lag."""

    edges = np.asarray(intervals, dtype=np.int64)
    return np.stack([
        (cumulative[edges[:, 1] + offset] - cumulative[edges[:, 0] + offset]).sum(axis=0)
        for offset in range(lag + 1)
    ])


def _shifted_products(signal, lag):
    """``Gamma[d][c1, c2] = sum_u z[u, c1] * z[u + d, c2]`` over the full grid."""

    rows = len(signal) - lag
    products = {}
    for shift in range(-lag, lag + 1):
        if shift >= 0:
            products[shift] = signal[:rows].T @ signal[shift : shift + rows]
        else:
            products[shift] = signal[-shift:rows].T @ signal[: rows + shift]
    return products


def whole_gram(signal, lag):
    """The design Gram of every row of a trial, without building the design."""

    channels = signal.shape[1]
    rows = len(signal) - lag
    products = _shifted_products(signal, lag)
    gram = np.empty((lag + 1, channels, lag + 1, channels))
    for first in range(lag + 1):
        for second in range(lag + 1):
            shift = second - first
            block = products[shift]
            # The identity's sum starts where both indices are valid; a window
            # grid starts at `first` and stops at `rows`, so both boundary
            # stretches are taken back out and the tail put back in.
            low = max(0, -shift)
            if first > low:
                block = block - signal[low:first].T @ signal[low + shift : first + shift]
            if first > 0:
                block = block + signal[rows : rows + first].T @ signal[
                    rows + shift : rows + first + shift
                ]
            gram[first, :, second, :] = block
    return gram.reshape(channels * (lag + 1), channels * (lag + 1))


def row_gram(signal, indices, lag):
    """The design Gram of an explicit row set: the rows a window grid drops."""

    channels = signal.shape[1]
    if not len(indices):
        width = channels * (lag + 1)
        return np.zeros((width, width))
    design = np.concatenate([signal[indices + offset] for offset in range(lag + 1)], axis=1)
    return design.T @ design


class Moments:
    """What one trial contributes to any fold, before normalization.

    Attributes:
        gram: raw design Gram ``sum_t x_t x_t^T``.
        design_sums: ``sum_t x_t``, shaped ``(lag + 1, channels)``.
        design_count: design rows behind both of the above.
        sample_sums / sample_squares / sample_count: moments of the window
            samples, which is what ``RidgeDecoder.fit`` normalizes with.
        target: ``sum_t x_t y_t`` with the per-window target normalization
            already applied, ``y`` being the attended envelope.
        target_sum: ``sum_t y_t``, for the mean correction.
        target_rows: rows behind ``target``.
    """

    __slots__ = ("gram", "design_sums", "design_count", "sample_sums",
                 "sample_squares", "sample_count", "target", "target_sum",
                 "target_rows")

    def __init__(self, gram, design_sums, design_count, sample_sums, sample_squares,
                 sample_count, target, target_sum, target_rows):
        self.gram = gram
        self.design_sums = design_sums
        self.design_count = int(design_count)
        self.sample_sums = sample_sums
        self.sample_squares = sample_squares
        self.sample_count = int(sample_count)
        self.target = target
        self.target_sum = float(target_sum)
        self.target_rows = int(target_rows)

    def __add__(self, other):
        return Moments(
            self.gram + other.gram, self.design_sums + other.design_sums,
            self.design_count + other.design_count,
            self.sample_sums + other.sample_sums,
            self.sample_squares + other.sample_squares,
            self.sample_count + other.sample_count,
            self.target + other.target, self.target_sum + other.target_sum,
            self.target_rows + other.target_rows,
        )

    def __sub__(self, other):
        if other.design_count > self.design_count:
            raise ValueError("Cannot remove more trials than the sum holds.")
        return Moments(
            self.gram - other.gram, self.design_sums - other.design_sums,
            self.design_count - other.design_count,
            self.sample_sums - other.sample_sums,
            self.sample_squares - other.sample_squares,
            self.sample_count - other.sample_count,
            self.target - other.target, self.target_sum - other.target_sum,
            self.target_rows - other.target_rows,
        )

    @property
    def width(self):
        return self.gram.shape[0]

    @property
    def channels(self):
        return len(self.sample_sums)


def empty(width, channels, lag):
    """A zero contribution, so a running sum has something well shaped."""

    return Moments(np.zeros((width, width)), np.zeros((lag + 1, channels)), 0,
                   np.zeros(channels), np.zeros(channels), 0,
                   np.zeros(width), 0.0, 0)


def trial_moments(signal, envelopes, label, starts, length, lag):
    """Measure one trial against one window grid.

    Windows whose attended envelope is constant are dropped, exactly as
    ``RidgeDecoder.fit`` drops them: they carry no target, so they must not
    enter the covariance either.
    """

    signal = np.asarray(signal, dtype=np.float64)
    envelopes = np.asarray(envelopes, dtype=np.float64)
    samples, channels = signal.shape
    kept = []
    # One target series for the whole trial, zero between windows: the lag-th
    # block of sum_t x_t y_t is then a single matrix-vector product on a shifted
    # slice, instead of one product per window per lag.
    series = np.zeros(samples - lag)
    target_sum = 0.0
    for start, stop in design_rows(starts, length, lag):
        if stop <= start:
            continue
        values = envelopes[start:stop, label]
        spread = values.std()
        if not np.isfinite(spread) or spread < 1e-12:
            continue
        kept.append((start, stop))
        normalized = (values - values.mean()) / spread
        series[start:stop] = normalized
        target_sum += float(normalized.sum())
    if not kept:
        return None
    target = np.concatenate([
        signal[offset : offset + samples - lag].T @ series
        for offset in range(lag + 1)
    ])
    excluded = np.ones(samples - lag, dtype=bool)
    for start, stop in kept:
        excluded[start:stop] = False
    gram = whole_gram(signal, lag) - row_gram(signal, np.flatnonzero(excluded), lag)
    cumulative = _cumulative(signal)
    windows = [(start, stop + lag) for start, stop in kept]
    return Moments(
        gram,
        _shifted_sums(cumulative, kept, lag),
        sum(stop - start for start, stop in kept),
        _interval_sums(cumulative, windows),
        _interval_sums(_cumulative(signal * signal), windows),
        sum(stop - start for start, stop in windows),
        target,
        target_sum,
        sum(stop - start for start, stop in kept),
    )


def normalize(moments):
    """The fold's per-channel mean and scale, from training windows only."""

    mean = moments.sample_sums / moments.sample_count
    variance = np.maximum(moments.sample_squares / moments.sample_count - mean * mean, 0)
    return mean, np.sqrt(variance)


def covariance(moments, mean, scale):
    """The normalized design covariance, exactly as ``fit`` accumulates it.

    ``sum (z - m)(z - m)^T = sum z z^T - m (sum z)^T - (sum z) m^T + n m m^T``,
    so the raw moments need only a rank-structured update.
    """

    channels = len(mean)
    lag = moments.width // channels - 1
    raw = moments.gram.reshape(lag + 1, channels, lag + 1, channels)
    sums = moments.design_sums.reshape(lag + 1, channels)
    outer = np.outer(mean, mean)
    adjusted = (
        raw
        - mean[None, :, None, None] * sums[None, None, :, :]
        - mean[None, None, None, :] * sums[:, :, None, None]
        + moments.design_count * outer[None, :, None, :]
    )
    adjusted = adjusted / (scale[None, :, None, None] * scale[None, None, None, :])
    return adjusted.reshape(moments.width, moments.width)


def fit(moments, contract, alpha=100.0, config=None, training_info=None):
    """Solve one fold and return a real, saved-shaped :class:`RidgeDecoder`."""

    config = AuditoryConfig() if config is None else config
    if moments.design_count == 0 or moments.target_rows == 0:
        raise ValueError("A fold needs at least one labeled training window.")
    mean, scale = normalize(moments)
    if np.any(scale < 1e-9):
        raise ValueError("Training contains a flat EEG channel.")
    lag = config.lag_samples
    if moments.width != len(mean) * (lag + 1):
        raise ValueError("Moments do not match the decoder's lag and channel count.")
    target = (moments.target - moments.target_sum * np.tile(mean, lag + 1)) / np.tile(
        scale, lag + 1
    )
    weights = np.linalg.solve(
        covariance(moments, mean, scale) + alpha * np.eye(moments.width), target
    )
    model = RidgeDecoder(config, alpha)
    model.mean = mean
    model.scale = scale
    model.weights = weights
    model.contract = contract
    model.training_info = dict(training_info or {})
    return model
