"""Opt-in software contract checks only; no model fitting or performance evaluation."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import numpy as np
from sklearn.linear_model import LogisticRegression

from scripts.dataproc.streaming.config import StreamConfig
from scripts.dataproc.streaming.interpolator import SampleInterpolator
from scripts.dataproc.streaming.processor import RunProcessor
from scripts.dataproc.streaming.tests import test_streaming
from scripts.dataproc.streaming.window import EEGWindow

from .adapter import WindowAdapter
from .covariance import CovarianceEstimator
from .model import RiemannModel
from .pipeline import RiemannPipeline
from .tangent import TangentSpace


def config() -> StreamConfig:
    """Return a small explicit software fixture, not amplifier settings."""

    return StreamConfig(
        500,
        0,
        "test",
        "none",
        stream_name="test",
        expected_channels=("F3", "F4"),
        eog_channels=(),
    )


class InterpolationChecks(unittest.TestCase):
    """Check repair limits and source order using known numeric fixtures."""

    def test_cross_chunk_repair_preserves_finite_values(self):
        interpolator = SampleInterpolator(config())
        data = np.array([[0.0, 0.0], [np.nan, 1.0], [2.0, 2.0]]) * 1e-6
        times = np.array([1.0, 1.002, 1.004])
        first = interpolator.feed((data[:2], times[:2]))
        self.assertEqual(len(first[0][0]), 1)
        self.assertEqual(len(interpolator.pending), 1)
        second = interpolator.feed((data[2:], times[2:]))
        np.testing.assert_allclose(second[0][0], [[1e-6, 1e-6], [2e-6, 2e-6]])
        self.assertEqual(interpolator.events[0]["available_at"], 1.004)
        self.assertTrue(np.isnan(data[1, 0]))

    def test_short_gap_and_long_fault(self):
        interpolator = SampleInterpolator(config())
        chunks = interpolator.feed(
            (np.array([[0.0, 0.0], [2.0, 2.0]]) * 1e-6, np.array([1.0, 1.004]))
        )
        np.testing.assert_allclose(chunks[0][0], [[0, 0], [1e-6, 1e-6], [2e-6, 2e-6]])
        self.assertEqual(interpolator.events[0]["samples"], 1)
        bad = np.full((20, 2), np.nan)
        chunks = interpolator.feed((bad, 1.006 + np.arange(20) / 500))
        self.assertFalse(interpolator.pending)
        self.assertTrue(np.isnan(chunks[0][0]).all())

    def test_no_extrapolation_or_saturated_endpoint(self):
        interpolator = SampleInterpolator(config())
        values = np.array([[np.nan, 0], [0, 0], [np.nan, 0], [1, 0]])
        chunks = interpolator.feed((values, 1 + np.arange(4) / 500))
        self.assertEqual(interpolator.events, [])
        self.assertTrue(any(np.isnan(data).any() for data, _ in chunks))

    def test_processor_repair_metadata_and_no_reset(self):
        settings = config()
        times = 10 + np.arange(4000) / 500
        values = (
            np.column_stack(
                (np.sin(2 * np.pi * 10 * times), np.cos(2 * np.pi * 10 * times))
            )
            * 20e-6
        )
        values[2600, 0] = np.nan
        processor = RunProcessor(settings)
        windows = []
        for start in range(0, len(values), 37):
            windows.extend(
                processor.feed((values[start : start + 37], times[start : start + 37]))
            )
        self.assertEqual(processor.recoveries, 0)
        self.assertEqual(processor.interpolated_samples, 1)
        self.assertTrue(any(window.interpolated for window in windows))
        self.assertTrue(
            all(window.contract == settings.window_contract() for window in windows)
        )


class ShapeChecks(unittest.TestCase):
    """Check array contracts and matrix utilities without fitting a classifier."""

    def test_explicit_layouts_and_order(self):
        adapter = WindowAdapter(("F3", "F4"), 256)
        data = np.arange(512).reshape(256, 2)
        window = EEGWindow(
            data,
            np.empty((256, 0)),
            np.arange(256),
            True,
            (),
            0,
            channel_names=("F3", "F4"),
        )
        self.assertEqual(adapter.transform(window).shape, (1, 256, 2))
        self.assertEqual(adapter.channels_first(data).shape, (1, 2, 256))
        self.assertEqual(adapter.channels_first(data, True).shape, (1, 1, 2, 256))
        window.channel_names = ("F4", "F3")
        with self.assertRaises(ValueError):
            adapter.transform(window)

    def test_regularized_rank_deficiency_and_fixed_reference(self):
        time = np.arange(256) / 128
        signal = np.sin(2 * np.pi * 10 * time)
        data = np.column_stack((signal, -signal))[None, :, :]
        covariance = CovarianceEstimator().transform(data)
        self.assertTrue(np.all(np.linalg.eigvalsh(covariance) > 0))
        tangent = TangentSpace()
        with self.assertRaises(RuntimeError):
            tangent.transform(covariance)
        tangent.set_reference(np.eye(2))
        np.testing.assert_allclose(tangent.transform(np.eye(2)[None, :, :]), 0)
        self.assertEqual(tangent.transform(covariance).shape, (1, 3))

    def test_pipeline_tubes_with_stub_classifier(self):
        """Verify tube flow and rejection gating, without training/evaluating a model."""

        settings = config()
        tangent = TangentSpace()
        tangent.set_reference(np.eye(2))
        classifier = SimpleNamespace(
            predict_proba=Mock(return_value=np.array([[0.25, 0.75]]))
        )
        model = cast(
            RiemannModel,
            SimpleNamespace(
                config=settings,
                validate_window=Mock(),
                covariance=CovarianceEstimator(),
                tangent=tangent,
                classifier=classifier,
                model_id="software-stub",
                classes=np.array([0, 1]),
            ),
        )
        pipeline = RiemannPipeline(model)
        times = np.arange(256) / 128
        data = np.column_stack(
            (np.sin(2 * np.pi * 10 * times), np.cos(2 * np.pi * 10 * times))
        )
        window = EEGWindow(
            data,
            np.empty((256, 0)),
            times,
            True,
            (),
            0,
            channel_names=settings.expected_channels,
            contract=settings.window_contract(),
        )
        prediction, original = pipeline.rundown(window)
        assert prediction is not None
        self.assertIs(original, window)
        self.assertEqual(prediction.probabilities, [0.25, 0.75])
        self.assertEqual(classifier.predict_proba.call_args.args[0].shape, (1, 3))
        window.valid = False
        window.reasons = ("warmup",)
        pipeline.feed(window)
        rejected, _ = pipeline.step(4)
        assert rejected is not None
        self.assertFalse(rejected.valid)
        self.assertIsNone(rejected.probabilities)
        self.assertEqual(classifier.predict_proba.call_count, 1)

    def test_numeric_bundle_roundtrip_without_fitting(self):
        """Check file persistence using manually supplied zero weights, not a trained model."""

        tangent = TangentSpace()
        tangent.set_reference(np.eye(2))
        classifier = LogisticRegression()
        classifier.classes_ = np.array([0, 1])
        classifier.coef_ = np.zeros((1, 3))
        classifier.intercept_ = np.zeros(1)
        vars(classifier)["n_features_in_"] = 3
        model = RiemannModel(config(), tangent, classifier, "software fixture only")
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.npz"
            model.save(path)
            restored = RiemannModel.load(path)
            self.assertEqual(restored.model_id, model.model_id)
            self.assertEqual(restored.contract, model.contract)
            np.testing.assert_array_equal(restored.classifier.coef_, classifier.coef_)
            with self.assertRaises(FileExistsError):
                model.save(path)

    def test_streamer_result_handoff_and_invalid_opt_in(self):
        """Use the fake inlet to check return-value delivery, not a trained model."""

        for include_invalid in (False, True):
            results = []
            streamer = test_streaming.LifecycleTests().runner(
                "data", pipeline=lambda window: ("stub", window)
            )
            streamer.pipeline_on_invalid = include_invalid
            streamer.on_result = results.append
            streamer.on_window = lambda window, runner=streamer: runner.stop()
            streamer.initialize()
            streamer.stream(duration=0.1)
            self.assertEqual(len(results), int(include_invalid))
            if include_invalid:
                self.assertEqual(results[0][0], "stub")
                self.assertFalse(results[0][1].valid)


if __name__ == "__main__":
    unittest.main()
