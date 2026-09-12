"""VoltageConverter: independent realtime component."""

import numpy as np


class VoltageConverter:
    """Convert canonical samples using preserved source unit exponents.

    Args:
        source_exponents: Voltage exponents in the same order as queued columns.
        desired_exponent: Output multiplier; -6 means microvolts.
    """

    def __init__(
        self, source_exponents: tuple[int, ...], desired_exponent: int = -6
    ) -> None:
        """Build per-channel multipliers from source units to microvolts."""

        self.converter = np.power(10.0, np.asarray(source_exponents) - desired_exponent)

    def convert(self, data: np.ndarray) -> np.ndarray:
        """Return converted samples without changing source data or metadata."""

        if data.ndim != 2 or data.shape[1] != len(self.converter):
            raise ValueError("Expected samples by channels matching the unit contract.")

        return data * self.converter[None, :]
