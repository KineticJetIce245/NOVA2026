"""Tests for the SoXR streaming resampler stage."""

import unittest
import warnings

import numpy as np

from nova2026.streaming.preprocess import (
    Resampler,
    ResamplerQualityWarning,
    select_quality,
)
from nova2026.streaming.preprocess.resample import (
    QUALITY_ORDER,
    _measure_startup_delay,
)

IN_SFREQ = 500.0
OUT_SFREQ = 128.0
CHANNELS = 4


def times(count: int, start: float = 0.0, sfreq: float = IN_SFREQ) -> np.ndarray:
    return start + np.arange(count) / sfreq


def feed_all(resampler: Resampler, data: np.ndarray, chunk_size: int):
    """Feed data in chunks of ``chunk_size`` and return the whole output."""

    pieces = []
    for start in range(0, len(data), chunk_size):
        block = data[start : start + chunk_size]
        out, _ = resampler(block, times(len(block), start=start / IN_SFREQ))
        pieces.append(out)
    return np.concatenate(pieces, axis=0)


class ResamplerTests(unittest.TestCase):
    def test_dc_signal_survives_unchanged(self) -> None:
        resampler = Resampler(IN_SFREQ, OUT_SFREQ, CHANNELS, quality="LQ")
        data = np.ones((6000, CHANNELS))

        output = feed_all(resampler, data, chunk_size=250)

        self.assertGreater(len(output), 0)
        # The very first output samples are a 0 -> 1 step transient while SoXR
        # fills its filter history; the settled tail must equal the DC input.
        tail = output[-512:]
        self.assertTrue(np.allclose(tail, 1.0, atol=1e-3))

    def test_output_rate_matches_the_ratio(self) -> None:
        resampler = Resampler(IN_SFREQ, OUT_SFREQ, CHANNELS, quality="LQ")
        rng = np.random.default_rng(0)
        count = 8000

        output = feed_all(resampler, rng.standard_normal((count, CHANNELS)), 200)

        # A streaming resampler retains its tail until more input arrives, so a
        # finite run under-reports the exact ratio by a bounded amount.
        expected = count * OUT_SFREQ / IN_SFREQ
        self.assertAlmostEqual(len(output) / expected, 1.0, delta=0.05)
        self.assertEqual(output.shape[1], CHANNELS)

    def test_timestamps_form_an_output_grid(self) -> None:
        resampler = Resampler(IN_SFREQ, OUT_SFREQ, CHANNELS, quality="LQ")
        pieces = []
        for start in range(0, 4000, 500):
            block = np.ones((500, CHANNELS))
            _, output_times = resampler(block, times(len(block), start=start / IN_SFREQ))
            pieces.append(output_times)
        joined = np.concatenate(pieces)

        self.assertTrue(np.all(np.diff(joined) > 0))
        self.assertTrue(
            np.allclose(np.diff(joined), 1.0 / OUT_SFREQ, atol=1e-9)
        )

    def test_reset_restarts_the_grid(self) -> None:
        resampler = Resampler(IN_SFREQ, OUT_SFREQ, CHANNELS, quality="LQ")
        feed_all(resampler, np.ones((1000, CHANNELS)), 500)

        resampler.reset()
        self.assertEqual(resampler.output_samples, 0)
        _, output_times = resampler(np.ones((6000, CHANNELS)), times(6000, start=10.0))
        self.assertGreater(len(output_times), 0)
        self.assertGreaterEqual(output_times[0], 10.0 - 1e-9)
        self.assertTrue(np.all(np.diff(output_times) > 0))

    def test_empty_chunk_is_a_noop(self) -> None:
        resampler = Resampler(IN_SFREQ, OUT_SFREQ, CHANNELS, quality="LQ")
        data, output_times = resampler(np.empty((0, CHANNELS)), np.empty(0))
        self.assertEqual(data.shape, (0, CHANNELS))
        self.assertEqual(output_times.shape, (0,))

    def test_validation(self) -> None:
        with self.assertRaises(ValueError):
            Resampler(0.0, OUT_SFREQ, CHANNELS)
        with self.assertRaises(ValueError):
            Resampler(IN_SFREQ, IN_SFREQ, CHANNELS)
        with self.assertRaises(ValueError):
            Resampler(IN_SFREQ, OUT_SFREQ, 0)
        with self.assertRaises(ValueError):
            Resampler(IN_SFREQ, OUT_SFREQ, CHANNELS, quality="ZZ")

        resampler = Resampler(IN_SFREQ, OUT_SFREQ, CHANNELS, quality="LQ")
        with self.assertRaises(ValueError):
            resampler(np.zeros((5, CHANNELS + 1)), times(5))
        with self.assertRaises(ValueError):
            resampler(np.zeros((5, CHANNELS)), times(4))


def measured_delays(in_sfreq=IN_SFREQ, out_sfreq=OUT_SFREQ) -> dict:
    """Measured startup delay of every anti-aliased preset on this build."""

    return {
        quality: _measure_startup_delay(in_sfreq, out_sfreq, quality)
        for quality in QUALITY_ORDER
    }


def expected_quality(budget: float, delays: dict) -> str:
    """The cleanest anti-aliased preset that fits ``budget`` (or the fastest)."""

    for quality in QUALITY_ORDER:
        if delays[quality] <= budget:
            return quality
    return min(delays, key=delays.get)


class AutomaticQualityTests(unittest.TestCase):
    """Automatic preset selection derives its budget from the allowed age."""

    def test_picks_the_cleanest_preset_that_fits(self) -> None:
        delays = measured_delays()
        budget = max(delays.values()) + 0.5  # everything fits
        resampler = Resampler(
            IN_SFREQ, OUT_SFREQ, CHANNELS, max_delay_seconds=budget
        )
        self.assertEqual(resampler.quality, QUALITY_ORDER[0])  # strongest
        self.assertLessEqual(resampler.startup_delay_seconds, budget)
        self.assertAlmostEqual(
            resampler.startup_delay_seconds, delays[resampler.quality], delta=1e-9
        )

    def test_tight_budget_selects_only_what_fits(self) -> None:
        delays = measured_delays()
        budget = delays["LQ"] + 0.01  # only the fastest clean preset fits
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResamplerQualityWarning)
            resampler = Resampler(
                IN_SFREQ, OUT_SFREQ, CHANNELS, max_delay_seconds=budget
            )
        self.assertEqual(resampler.quality, expected_quality(budget, delays))
        self.assertLessEqual(resampler.startup_delay_seconds, budget)

    def test_budget_derived_from_allowed_age_and_reserve(self) -> None:
        delays = measured_delays()
        reserve = 0.25
        budget = delays["LQ"] + 0.2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResamplerQualityWarning)
            resampler = Resampler(
                IN_SFREQ, OUT_SFREQ, CHANNELS,
                max_age_seconds=budget + reserve, reserve_seconds=reserve,
            )
        self.assertAlmostEqual(resampler.max_delay_seconds, budget, delta=1e-9)
        self.assertEqual(resampler.quality, expected_quality(budget, delays))

    def test_reserve_larger_than_allowed_age_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "reserve_seconds"):
            Resampler(
                IN_SFREQ, OUT_SFREQ, CHANNELS,
                max_age_seconds=1.0, reserve_seconds=1.5,
            )

    def test_unreachable_budget_warns_and_keeps_the_fastest(self) -> None:
        delays = measured_delays()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resampler = Resampler(
                IN_SFREQ, OUT_SFREQ, CHANNELS, max_delay_seconds=0.001
            )
        self.assertEqual(resampler.quality, min(delays, key=delays.get))
        self.assertTrue(any("delay budget" in str(w.message) for w in caught))
        self.assertTrue(
            all(issubclass(w.category, ResamplerQualityWarning) for w in caught)
        )

    def test_strict_raises_instead_of_warning(self) -> None:
        with self.assertRaisesRegex(ValueError, "delay budget"):
            Resampler(
                IN_SFREQ, OUT_SFREQ, CHANNELS,
                max_delay_seconds=0.001, strict=True,
            )

    def test_qq_is_opt_in(self) -> None:
        with self.assertRaisesRegex(ValueError, "QQ"):
            Resampler(IN_SFREQ, OUT_SFREQ, CHANNELS, quality="QQ")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # QQ needs one input sample (0.002 s here); 0.01 s is enough.
            resampler = Resampler(
                IN_SFREQ, OUT_SFREQ, CHANNELS,
                max_delay_seconds=0.01, allow_qq=True,
            )
        self.assertEqual(resampler.quality, "QQ")
        self.assertTrue(any("anti-aliasing" in str(w.message) for w in caught))

    def test_explicit_qq_is_accepted_with_opt_in(self) -> None:
        # Documented behaviour: naming QQ directly works once allow_qq=True.
        resampler = Resampler(
            IN_SFREQ, OUT_SFREQ, CHANNELS, quality="QQ", allow_qq=True
        )
        self.assertEqual(resampler.quality, "QQ")
        output, _ = resampler(np.ones((500, CHANNELS)), times(500))
        self.assertGreater(len(output), 0)  # QQ emits from the first chunk

    def test_auto_spelling_and_explicit_choice(self) -> None:
        delays = measured_delays()
        budget = max(delays.values()) + 0.5
        resampler = Resampler(
            IN_SFREQ, OUT_SFREQ, CHANNELS, "auto", max_delay_seconds=budget
        )
        self.assertEqual(resampler.quality, QUALITY_ORDER[0])
        # An explicit preset is the caller's choice: no quality warning.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            explicit = Resampler(IN_SFREQ, OUT_SFREQ, CHANNELS, quality="LQ")
        self.assertEqual(explicit.quality, "LQ")
        self.assertEqual(caught, [])

    def test_option_validation(self) -> None:
        with self.assertRaises(ValueError):
            Resampler(IN_SFREQ, OUT_SFREQ, CHANNELS, max_delay_seconds=0.0)
        with self.assertRaises(ValueError):
            Resampler(IN_SFREQ, OUT_SFREQ, CHANNELS, max_age_seconds=float("nan"))
        with self.assertRaises(ValueError):
            Resampler(IN_SFREQ, OUT_SFREQ, CHANNELS, reserve_seconds=-1.0)
        with self.assertRaises(TypeError):
            Resampler(IN_SFREQ, OUT_SFREQ, CHANNELS, allow_qq=1)
        with self.assertRaises(TypeError):
            Resampler(IN_SFREQ, OUT_SFREQ, CHANNELS, strict=1)

    def test_select_quality_is_public_and_consistent(self) -> None:
        quality, delay = select_quality(
            IN_SFREQ, OUT_SFREQ, max_delay_seconds=0.001
        )
        self.assertIn(quality, QUALITY_ORDER)
        self.assertGreater(delay, 0.001)
        with self.assertRaises(ValueError):
            select_quality(IN_SFREQ, OUT_SFREQ, max_delay_seconds=-1.0)


if __name__ == "__main__":
    unittest.main()
