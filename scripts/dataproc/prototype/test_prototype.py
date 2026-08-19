"""Synthetic unit tests for the prototype AttentivU implementation."""

import unittest

import mne
import numpy as np

from nova2026.config import SAMPLE_RATE

from .pipelines import attentivu_pipeline, center_scale_clip_channel
from .spectral import compute_trial_spectral_features


class SpectralFeatureTests(unittest.TestCase):
    def test_psd_and_features_preserve_trial_and_channel_dimensions(self) -> None:
        duration_seconds = 2
        sample_count = SAMPLE_RATE * duration_seconds
        times = np.arange(sample_count) / SAMPLE_RATE

        windows = np.empty((2, 2, sample_count), dtype=np.float64)
        windows[0, 0] = np.sin(2 * np.pi * 15 * times)
        windows[0, 1] = np.sin(2 * np.pi * 6 * times)
        windows[1, 0] = np.sin(2 * np.pi * 6 * times)
        windows[1, 1] = np.sin(2 * np.pi * 15 * times)

        features = compute_trial_spectral_features(windows, SAMPLE_RATE)

        self.assertEqual(features["psd"].shape[:2], (2, 2))
        self.assertEqual(features["theta"].shape, (2, 2))
        self.assertEqual(features["alpha"].shape, (2, 2))
        self.assertEqual(features["beta"].shape, (2, 2))
        self.assertEqual(features["engagement"].shape, (2, 2))
        self.assertGreater(features["engagement"][0, 0], features["engagement"][0, 1])
        self.assertGreater(features["engagement"][1, 1], features["engagement"][1, 0])

    def test_zero_power_produces_finite_engagement(self) -> None:
        windows = np.zeros((3, 4, SAMPLE_RATE * 2), dtype=np.float64)
        features = compute_trial_spectral_features(windows, SAMPLE_RATE)

        self.assertTrue(np.isfinite(features["engagement"]).all())
        np.testing.assert_array_equal(features["engagement"], 0.0)

    def test_invalid_window_dimensions_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            compute_trial_spectral_features(
                np.zeros((2, SAMPLE_RATE * 2)),
                SAMPLE_RATE,
            )


class PipelineTests(unittest.TestCase):
    def test_center_scale_clip_channel_is_bounded(self) -> None:
        values_v = np.array([-80.0, 0.0, 80.0]) * 1e-6
        normalized_v = center_scale_clip_channel(values_v)
        normalized_uv = normalized_v * 1e6

        self.assertGreaterEqual(float(normalized_uv.min()), -4.0)
        self.assertLessEqual(float(normalized_uv.max()), 4.0)

    def test_attentivu_pipeline_resamples_and_keeps_channels_independent(self) -> None:
        original_sfreq = 500.0
        duration_seconds = 5
        times = np.arange(int(original_sfreq * duration_seconds)) / original_sfreq
        data_v = np.vstack(
            [
                20e-6 * np.sin(2 * np.pi * 10 * times),
                20e-6 * np.sin(2 * np.pi * 15 * times),
            ]
        )
        info = mne.create_info(
            ch_names=["EEG-1", "EEG-2"],
            sfreq=original_sfreq,
            ch_types=["eeg", "eeg"],
        )
        raw = mne.io.RawArray(data_v, info, verbose=False)

        attentivu_pipeline(raw)

        self.assertEqual(raw.info["sfreq"], SAMPLE_RATE)
        output_uv = raw.get_data() * 1e6
        self.assertGreaterEqual(float(output_uv.min()), -4.0)
        self.assertLessEqual(float(output_uv.max()), 4.0)
        self.assertFalse(np.array_equal(output_uv[0], output_uv[1]))


if __name__ == "__main__":
    unittest.main()
