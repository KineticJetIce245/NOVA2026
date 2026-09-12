"""EEGNet variant for short post-stimulus windows.

Why a local copy instead of ``nova2026.architecture.cnn.EEGNet``
----------------------------------------------------------------
The shared EEGNet hard-codes its geometry for a 2 s window:

* the temporal kernel is ``int(fs * 0.5)`` samples, i.e. 500 ms at 128 Hz, so
  it cannot be shortened without misdeclaring the sampling rate;
* the pooling is fixed at ``(1, 4)`` then ``(1, 8)``, a 32x temporal reduction,
  which collapses a 26-sample (100-300 ms at 128 Hz) window to zero width.

This class parameterises both (``kernel_ms``, ``pool1``, ``pool2``) and keeps
the same block structure, so results stay comparable with the pre-stimulus arm.
It is meant to move into ``nova2026/architecture/cnn.py`` once the winning
geometry is settled.
"""

from __future__ import annotations

import torch
import torch.nn as nn

CLASSES = 2
TEMPORAL_FILTER = 8
DEPTHWISE_FILTER = 2
DROP_OUT_RATE = 0.5
SAMPLE_RATE = 128


def _pooled_length(length: int, kernel: int) -> int:
    """Output length of ``nn.AvgPool2d(kernel_size=(1, kernel))`` on ``length``."""
    if length < kernel:
        raise ValueError(
            f"Pooling with kernel {kernel} would erase a {length}-sample window; "
            "shorten the kernel/pool sizes or widen --window."
        )
    return length // kernel


class EEGNetWindow(nn.Module):
    """EEGNet (Lawhern et al., 2018) sized by time in milliseconds.

    Parameters
    ----------
    chn
        Number of EEG channels in the input.
    fs
        Sampling rate the kernel lengths are expressed against.
    t
        Number of time samples per epoch.
    kernel_ms
        Temporal convolution kernel length in milliseconds.
    f1, d, p
        As in EEGNet: temporal filters, depth multiplier, dropout.
    pool1, pool2
        Temporal pooling kernel after block 1 / block 2. ``pool2=None`` skips
        the second pooling, which is what keeps very short windows alive.
    classes
        Number of output logits.
    """

    def __init__(
        self,
        chn: int,
        fs: float = SAMPLE_RATE,
        t: int = 26,
        kernel_ms: float = 125.0,
        f1: int = TEMPORAL_FILTER,
        d: int = DEPTHWISE_FILTER,
        p: float = DROP_OUT_RATE,
        pool1: int = 2,
        pool2: int | None = 4,
        classes: int = CLASSES,
    ) -> None:
        super().__init__()
        if chn < 1 or t < 1:
            raise ValueError(f"Invalid input geometry (chn={chn}, t={t}).")
        kernel = max(1, int(round(fs * kernel_ms / 1000.0)))
        sep_kernel = max(1, int(round(fs * 0.125)))
        self.input_shape = (chn, t)
        self.kernel = kernel

        # Block 1 ==========================================================
        self.temporal_conv = nn.Conv2d(
            1, f1, kernel_size=(1, kernel), padding="same", bias=False
        )
        self.depthwise_conv = nn.Conv2d(
            f1, f1 * d, kernel_size=(chn, 1), groups=f1, padding="valid", bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(f1)
        self.batchnorm2 = nn.BatchNorm2d(f1 * d)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, pool1)) if pool1 else None
        t1 = _pooled_length(t, pool1) if pool1 else t

        # Block 2 ==========================================================
        self.sep_depthwise_conv = nn.Conv2d(
            f1 * d,
            f1 * d,
            kernel_size=(1, sep_kernel),
            groups=f1 * d,
            padding="same",
            bias=False,
        )
        self.sep_pointwise_conv = nn.Conv2d(
            f1 * d, f1 * d, kernel_size=(1, 1), bias=False
        )
        self.batchnorm3 = nn.BatchNorm2d(f1 * d)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, pool2)) if pool2 else None
        t2 = _pooled_length(t1, pool2) if pool2 else t1

        self.classifier = nn.LazyLinear(classes)
        self.features = f1 * d * t2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, C, T)
        if tuple(x.shape[-2:]) != self.input_shape:
            raise ValueError(
                f"EEGNetWindow expects (C, T)={self.input_shape}, got "
                f"{tuple(x.shape[-2:])}."
            )

        x = self.batchnorm1(self.temporal_conv(x))
        x = self.batchnorm2(self.depthwise_conv(x))
        x = self.elu(x)
        if self.pool1 is not None:
            x = self.pool1(x)
        x = self.dropout(x)

        x = self.sep_depthwise_conv(x)
        x = self.sep_pointwise_conv(x)
        x = self.elu(self.batchnorm3(x))
        if self.pool2 is not None:
            x = self.pool2(x)
        x = self.dropout(x)

        return self.classifier(x.flatten(start_dim=1))
