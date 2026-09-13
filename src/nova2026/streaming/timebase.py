"""One owner for the LSL timeline: place received samples on a regular grid.

The amplifier's timestamps cannot be trusted for placement: on the 2026-09-12
bring-up 0.48% of the steps were shorter than half a sample, which ``Repair``
refuses at any tolerance, so something has to turn those stamps into a grid
``Repair`` accepts. The design question is what that something is allowed to do.
``documents/timebase_design.md`` carries the measurements; the short version:

* the recorded samples match the amplifier's own ``.cnt`` sample for sample
  (88 500 pairs, ``r = 1.000000``, nothing differing by more than one LSB), so
  **the data is intact and the timeline is what is wrong**;
* the timeline's error is a staircase of whole-sample steps plus a few hundred
  ppm of rate error (2-9 samples over 30 s), not a rate error alone;
* the old code read those steps as *lost samples* and inserted phantom slots for
  them. ``Repair`` then "repaired" the rows that never existed, which flagged
  windows ``interpolated`` and, on a DC-offset signal, raised
  ``unsafe_endpoints``. **The pipeline fabricated damage in response to a
  timekeeping artefact.**

So this module has one invariant and one habit:

* **one grid slot per sample actually received.** A timeline anomaly may change
  where the next sample lands; it may never change how many rows there are.
* **report, do not hide.** The disagreement between the anchors and the sample
  count, every suspicious step, and every correction are returned as data, so
  the run's report and scoring rules can judge them. This module does not decide
  that samples were lost, because from timestamps alone that cannot be decided:
  a clock correction and a genuine loss of *k* samples both appear as a
  permanent *+k* step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .preprocess.repair import grid_tolerance_samples

# The grid tolerance is not this module's to define: it belongs to the stage that
# applies it. ``grid_tolerance_samples`` is imported from ``Repair`` and used
# here, so the timeline can never be held to a rule the consumer will refuse.


@dataclass(frozen=True)
class GridPolicy:
    """What a time base may do to a source whose stamps cannot be trusted.

    Args:
        nominal_sfreq: The rate the grid is laid at, in Hz. It is the declared
            rate, deliberately: the model contract and the window geometry are
            defined against it, and the measurements say it is good to better
            than a few hundred ppm.
        tolerance_samples: Jitter absorbed without comment, in samples. Defaults
            to whatever ``Repair`` is given at this rate, so the time base cannot
            accept a grid the consumer will refuse.
        relock_samples: Disagreement tolerated before the grid re-anchors.
        relock_slew_samples: Floor for how many samples a re-lock is spread over.
            The actual count is raised when needed so no single step leaves the
            tolerance (see :meth:`TimeBase._start_relock`).
        max_step_samples: A single anchor step at or above this many samples is
            recorded as a suspicious jump. It is never treated as loss.

    Raises:
        ValueError: On a non-positive rate, a tolerance outside ``(0, 0.5)``
            samples, a non-positive re-lock threshold, a slew below one sample,
            or a step threshold at or below one sample.
    """

    nominal_sfreq: float
    tolerance_samples: float | None = None
    relock_samples: float = 0.5
    relock_slew_samples: int = 50
    max_step_samples: float = 1.5

    def __post_init__(self) -> None:
        """Validate the policy and fill in the shared tolerance."""

        if not math.isfinite(self.nominal_sfreq) or self.nominal_sfreq <= 0:
            raise ValueError("nominal_sfreq must be finite and positive.")
        tolerance = self.tolerance_samples
        if tolerance is None:
            tolerance = grid_tolerance_samples(self.nominal_sfreq)
            object.__setattr__(self, "tolerance_samples", tolerance)
        if not math.isfinite(tolerance) or not 0.0 < tolerance < 0.5:
            # Repair itself refuses a tolerance at or above half a sample, so a
            # policy that asks for one could never be satisfied.
            raise ValueError("tolerance_samples must be inside (0, 0.5) samples.")
        if not math.isfinite(self.relock_samples) or self.relock_samples <= 0:
            raise ValueError("relock_samples must be finite and positive.")
        if self.relock_slew_samples < 1:
            raise ValueError("relock_slew_samples must be at least 1.")
        if not math.isfinite(self.max_step_samples) or self.max_step_samples <= 1.0:
            raise ValueError("max_step_samples must be finite and above 1 sample.")

    @property
    def tolerance_seconds(self) -> float:
        """The same tolerance in the units ``Repair`` takes."""

        return float(self.tolerance_samples) / self.nominal_sfreq

    @classmethod
    def for_rate(cls, sfreq: float, **overrides) -> GridPolicy:
        """The policy the live chain uses, with every field overridable."""

        return cls(nominal_sfreq=sfreq, **overrides)


@dataclass(frozen=True)
class TimeBaseEvent:
    """One thing that happened to the timeline, kept so a run can report it.

    Args:
        kind: ``"relock"`` (the grid re-anchored) or ``"large_step"`` (a single
            anchor step the source produced).
        index: Sample index at which it happened.
        size_samples: Sign and size in samples: for a re-lock the disagreement
            that was absorbed, for a step the step itself.
        slew_samples: Samples the re-lock was spread over; 0 for a step.
    """

    kind: str
    index: int
    size_samples: float
    slew_samples: int = 0


@dataclass(frozen=True)
class TimeBaseState:
    """What a run has to say about its own timeline.

    Args:
        total_samples: Slots placed so far - always the number of samples
            received, never a fabricated one.
        residual_samples: Disagreement right now, in samples.
        residual_peak_samples: Worst disagreement seen at a chunk boundary,
            before any re-lock absorbed it.
        anchor_rate: Rate the anchors themselves imply, in Hz (``nan`` until two
            anchors are known). Compare it with ``nominal_sfreq``: the difference
            is the ppm error of the source's clock.
        relocks: Every re-anchor, in order.
        large_steps: Every suspicious single step, in order.
    """

    total_samples: int
    residual_samples: float
    residual_peak_samples: float
    anchor_rate: float
    relocks: tuple[TimeBaseEvent, ...] = field(default_factory=tuple)
    large_steps: tuple[TimeBaseEvent, ...] = field(default_factory=tuple)

    @property
    def relocked_samples(self) -> float:
        """Total drift this run absorbed by re-locking, in samples.

        This, not ``residual_peak_samples``, is the number to score: the peak is
        bounded by the policy by construction, so it cannot show how far the
        source's clock and the grid disagreed over the session.
        """

        return float(sum(abs(event.size_samples) for event in self.relocks))


class TimeBase:
    """Place every received sample on a regular grid at the nominal rate.

    The grid is counted, not stamped: sample ``n`` lands at
    ``anchor + n / nominal_sfreq``. Source stamps are read for one thing only -
    to measure how far the anchors have drifted from that count, which is
    reported and, past ``relock_samples``, corrected by slewing the anchor back
    over a number of samples chosen so the correction never breaks the consumer's
    grid tolerance.

    Args:
        policy: The :class:`GridPolicy` to obey.
        anchor: Timestamp of the first sample, when the caller already knows it.
            Otherwise the first stamp seen anchors the grid.

    Raises:
        TypeError: When ``policy`` is not a :class:`GridPolicy`.
        ValueError: When ``anchor`` is given and is not finite.
    """

    def __init__(self, policy: GridPolicy, *, anchor: float | None = None) -> None:
        """Validate the settings and start with no samples placed."""

        if not isinstance(policy, GridPolicy):
            raise TypeError("policy must be a GridPolicy.")
        if anchor is not None and not math.isfinite(anchor):
            raise ValueError("anchor must be finite when given.")
        self.policy = policy
        self._rate = policy.nominal_sfreq
        self._anchor = None if anchor is None else float(anchor)
        self._first_stamp = self._anchor
        self._last_stamp = None
        self._next_index = 0
        self._shift = 0.0
        self._shift_step = 0.0
        self._slew_left = 0
        self._residual = 0.0
        self._residual_peak = 0.0
        self._relocks: list[TimeBaseEvent] = []
        self._large_steps: list[TimeBaseEvent] = []

    @property
    def state(self) -> TimeBaseState:
        """The timeline as the run report should carry it."""

        span = (
            0.0
            if self._first_stamp is None or self._last_stamp is None
            else self._last_stamp - self._first_stamp
        )
        rate = float("nan")
        if span > 0 and self._next_index > 1:
            rate = (self._next_index - 1) / span
        return TimeBaseState(
            total_samples=self._next_index,
            residual_samples=self._residual,
            residual_peak_samples=self._residual_peak,
            anchor_rate=rate,
            relocks=tuple(self._relocks),
            large_steps=tuple(self._large_steps),
        )

    def place(
        self, stamps: np.ndarray, n_samples: int | None = None
    ) -> tuple[np.ndarray, TimeBaseState]:
        """Give one pulled chunk its place on the grid.

        Args:
            stamps: Timestamps of the chunk's samples, oldest first. A source
                that stamps a whole chunk once may pass a single anchor instead,
                as long as ``n_samples`` says how many samples it covers.
            n_samples: Samples the chunk carries. Defaults to ``len(stamps)``.

        Returns:
            ``(timestamps, state)``: one timestamp per received sample, plus the
            timeline's state after this chunk.

        Raises:
            ValueError: On empty or non-finite stamps, on a shape that is neither
                one stamp per sample nor a single chunk anchor, or on a
                non-positive ``n_samples``.
        """

        stamps, count, per_sample = self._read_chunk(stamps, n_samples)
        if count == 0:
            return np.empty(0), self.state

        anchor_stamp = float(stamps[0])
        if self._anchor is None:
            self._anchor = anchor_stamp
        if self._first_stamp is None:
            self._first_stamp = anchor_stamp
        if per_sample:
            self._record_large_steps(stamps)

        residual = (anchor_stamp - self._anchor) * self._rate - (
            self._next_index + self._shift
        )
        self._residual = residual
        self._residual_peak = max(self._residual_peak, abs(residual))
        if abs(residual) > self.policy.relock_samples:
            self._start_relock(residual)

        times = self._emit(count)
        self._next_index += count
        self._last_stamp = float(stamps[-1])
        return times, self.state

    def _read_chunk(
        self, stamps, n_samples: int | None
    ) -> tuple[np.ndarray, int, bool]:
        """Validate one chunk and report how many samples it carries.

        Returns ``(stamps, count, per_sample)``, where ``per_sample`` says whether
        the stamps describe every sample or just the chunk's anchor.
        """

        stamps = np.asarray(stamps, dtype=float)
        if stamps.ndim != 1:
            raise ValueError("stamps must be a 1-D array.")
        if stamps.size == 0:
            return stamps, 0, True
        if not np.isfinite(stamps).all():
            raise ValueError("stamps must be finite.")
        if n_samples is None:
            return stamps, int(stamps.size), True
        count = int(n_samples)
        if count < 1:
            raise ValueError("n_samples must be positive.")
        per_sample = stamps.size == count
        if not per_sample and stamps.size != 1:
            raise ValueError(
                "pass one timestamp per sample, or a single chunk anchor."
            )
        return stamps, count, per_sample

    def _record_large_steps(self, stamps: np.ndarray) -> None:
        """Record the suspicious steps in one chunk, the seam included.

        A step where the source's clock jumped is evidence about that clock, not
        proof that samples were lost: it is reported, and every sample that did
        arrive is still placed.

        The step between the previous chunk's last sample and this chunk's first
        one counts too. A re-stamp that happens to land on a block boundary is no
        different from one inside a block, and leaving the seam out would report
        fewer jumps purely because the source delivers data in blocks.
        """

        if self._last_stamp is not None:
            seam = (float(stamps[0]) - self._last_stamp) * self._rate
            if seam >= self.policy.max_step_samples:
                self._large_steps.append(
                    TimeBaseEvent("large_step", self._next_index, float(seam))
                )
        if stamps.size < 2:
            return
        steps = np.diff(stamps) * self._rate
        for offset in np.flatnonzero(steps >= self.policy.max_step_samples):
            self._large_steps.append(
                TimeBaseEvent(
                    "large_step",
                    self._next_index + int(offset),
                    float(steps[offset]),
                )
            )

    def _start_relock(self, residual: float) -> None:
        """Spread a correction of ``residual`` samples over the next samples."""

        # Each slewed sample moves by residual / slew, so the floor is raised
        # whenever half the consumer's tolerance would be exceeded: a re-lock
        # must not break the very grid rule this class exists to satisfy.
        half_tolerance = 0.5 * float(self.policy.tolerance_samples)
        needed = int(math.ceil(abs(residual) / half_tolerance))
        slew = max(self.policy.relock_slew_samples, needed)
        self._shift_step = residual / slew
        self._slew_left = slew
        self._relocks.append(
            TimeBaseEvent("relock", self._next_index, residual, slew)
        )

    def _emit(self, count: int) -> np.ndarray:
        """Return ``count`` timestamps, applying any re-lock in progress."""

        if self._slew_left <= 0:
            index = self._next_index + self._shift + np.arange(count)
            return self._anchor + index / self._rate
        times = np.empty(count)
        shift = self._shift
        left = self._slew_left
        for offset in range(count):
            times[offset] = (
                self._anchor + (self._next_index + offset + shift) / self._rate
            )
            if left > 0:
                shift += self._shift_step
                left -= 1
        self._shift = shift
        self._slew_left = left
        if left == 0:
            self._shift_step = 0.0
        return times
