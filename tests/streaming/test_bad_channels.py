"""Dry-cap regression tests: one dead electrode must not stop a live run.

Both kill paths are covered, because both used to collapse the channel
dimension before deciding anything:

1. The quality monitor rejects every window a persistent fault overlaps, and
   Recovery stops the run after ``persistent_fault_seconds`` of badness.
2. Repair cannot bridge an endless non-finite run, and refuses to interpolate
   between extreme endpoints, so one bad column is enough to exhaust the
   bounded recovery budget.
"""

import unittest
from types import SimpleNamespace

import numpy as np

from nova2026.auditory.config import AuditoryConfig
from nova2026.auditory.streaming import AuditoryProcessor, stream_config

N_CHANNELS = 8
SFREQ = 500.0
NAMES = tuple(f"E{index + 1:02d}" for index in range(N_CHANNELS))
CHUNK = 250  # 0.5 s


def trial() -> SimpleNamespace:
    """The source metadata stream_config reads."""

    return SimpleNamespace(
        sample_rate=SFREQ,
        channel_names=NAMES,
        reference="CPz",
        upstream_processing="none",
    )


def dry_cap(seconds: float, *, flat=(), missing=(), missing_after=None):
    """Synthetic dry-cap data: noisy live electrodes plus dead ones.

    ``flat`` columns deliver a constant (a disconnected electrode), ``missing``
    columns deliver no value at all (a column that never becomes finite), and
    ``missing_after=(index, seconds)`` drops one column from that time onward.
    """

    count = round(seconds * SFREQ)
    stamps = np.arange(count) / SFREQ
    rng = np.random.default_rng(0)
    data = 20.0 * np.sin(2 * np.pi * 10 * stamps)[:, None] * np.ones((1, N_CHANNELS))
    data = data + rng.normal(0, 2.0, data.shape)
    for index in flat:
        data[:, index] = 0.0
    for index in missing:
        data[:, index] = np.nan
    if missing_after is not None:
        index, after = missing_after
        data[round(after * SFREQ) :, index] = np.nan

    return data, stamps


def feed(processor: AuditoryProcessor, data: np.ndarray, stamps: np.ndarray) -> list:
    """Push the whole recording through the real chain in 0.5 s chunks."""

    windows = []
    for start in range(0, len(data), CHUNK):
        windows.extend(
            processor.feed((data[start : start + CHUNK], stamps[start : start + CHUNK]))
        )

    return windows


class DryCapTests(unittest.TestCase):
    def processor(self, judges=None, **options) -> AuditoryProcessor:
        settings = stream_config(trial(), AuditoryConfig(), 5.0, 1.0, **options)
        return AuditoryProcessor(settings, judges=judges)

    def test_default_policy_stops_the_run_on_one_dead_channel(self) -> None:
        """The behaviour the dry cap used to hit: 15 s, then the run ends."""

        processor = self.processor()
        data, stamps = dry_cap(20.0, flat=(0,))

        with self.assertRaisesRegex(RuntimeError, "quality faults persisted"):
            feed(processor, data, stamps)

    def test_excluded_dead_channel_keeps_streaming_and_is_recorded(self) -> None:
        processor = self.processor(exclude_channels=("E01",))
        data, stamps = dry_cap(20.0, flat=(0,))

        windows = feed(processor, data, stamps)
        valid = [window for window in windows if window.valid]

        self.assertTrue(valid, "no window survived after the first 20 s")
        self.assertTrue(all(window.bad_channels == ("E01",) for window in valid))
        self.assertEqual(processor.recovery.recoveries, 0)
        self.assertEqual(processor.recovery.segment, 0)

    def test_channel_budget_tolerates_a_census(self) -> None:
        processor = self.processor(max_bad_channels=3)
        data, stamps = dry_cap(20.0, flat=(0, 1, 4))

        valid = [window for window in feed(processor, data, stamps) if window.valid]

        self.assertTrue(valid)
        self.assertEqual(valid[-1].bad_channels, ("E01", "E02", "E05"))
        self.assertEqual(processor.recovery.recoveries, 0)

    def test_channel_budget_below_the_census_still_stops(self) -> None:
        processor = self.processor(max_bad_channels=2)
        data, stamps = dry_cap(20.0, flat=(0, 1, 4))

        with self.assertRaisesRegex(RuntimeError, "quality faults persisted"):
            feed(processor, data, stamps)

    def test_excluded_channel_that_never_reports_keeps_streaming(self) -> None:
        processor = self.processor(exclude_channels=("E03",))
        data, stamps = dry_cap(20.0, missing=(2,))

        windows = feed(processor, data, stamps)
        valid = [window for window in windows if window.valid]

        self.assertTrue(valid, "the held column still stopped the run")
        self.assertIn("E03", valid[-1].bad_channels)
        self.assertGreater(processor.repair.held_rows, 0)
        self.assertEqual(processor.repair.repaired_samples, 0)

    def test_excluded_channel_that_drops_mid_run_keeps_streaming(self) -> None:
        processor = self.processor(exclude_channels=("E03",))
        data, stamps = dry_cap(20.0, missing_after=(2, 0.5))

        valid = [window for window in feed(processor, data, stamps) if window.valid]

        self.assertTrue(valid)
        self.assertGreater(processor.repair.held_rows, 0)
        self.assertEqual(processor.recovery.recoveries, 0)

    def test_unscoped_channel_that_never_reports_starves_the_run(self) -> None:
        """Damage before the first finite row is dropped, not repaired.

        Nothing reaches the buffer, so the run produces no window at all
        instead of an error: ``live`` turns that into "No EEG decisions
        produced" at the end. Locked here so the silent shape stays visible.
        """

        processor = self.processor()
        data, stamps = dry_cap(4.0, missing=(2,))

        windows = feed(processor, data, stamps)

        self.assertEqual(windows, [])
        self.assertGreater(processor.repair.dropped_rows, 0)
        self.assertEqual(processor.recovery.recoveries, 0)

    def test_unscoped_channel_that_drops_mid_run_starves_after_one_recovery(self) -> None:
        """A lost column costs one bounded recovery, then the run starves.

        The first unrepairable chunk restarts the chain; every later chunk is
        then damage before the first finite row, which Repair drops. No window
        is produced and no further error is raised, which is exactly why the
        dead column has to be scoped out instead of tolerated by luck.
        """

        processor = self.processor()
        data, stamps = dry_cap(4.0, missing_after=(2, 0.5))

        windows = feed(processor, data, stamps)

        self.assertEqual(windows, [])
        self.assertEqual(processor.recovery.recoveries, 1)
        self.assertEqual(processor.recovery.segment, 1)
        self.assertGreater(processor.repair.dropped_rows, 0)

    def test_policy_is_not_part_of_the_model_contract(self) -> None:
        """Contract equality gates every existing decoder; keep it stable."""

        plain = self.processor()
        scoped = self.processor(
            check_channels=False, max_bad_channels=3, exclude_channels=("E01",)
        )

        self.assertEqual(plain.contract, scoped.contract)

    def test_injected_judge_is_used_instead_of_the_defaults(self) -> None:
        class Judge:
            """A census judge that sees one specific electrode as bad."""

            def reasons(self, start, end):
                return ()

            def bad_channels(self, start, end):
                return ("E07",)

        processor = self.processor(judges=(Judge(),))
        data, stamps = dry_cap(8.0)

        valid = [window for window in feed(processor, data, stamps) if window.valid]

        self.assertTrue(valid)
        # The injected judge replaced the defaults: no fault is reported at all.
        self.assertTrue(all(window.bad_channels == ("E07",) for window in valid))

    def test_injected_judge_must_implement_reasons(self) -> None:
        with self.assertRaisesRegex(TypeError, "reasons"):
            self.processor(judges=(object(),))
        with self.assertRaisesRegex(TypeError, "iterable"):
            self.processor(judges="not-a-judge")


if __name__ == "__main__":
    unittest.main()
