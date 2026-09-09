"""Tests for spatial projection (C): contract, fit_ssp, SpatialOperator."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from nova2026.streaming import (
    SpatialOperator,
    EEGWindow,
    cut_epochs,
    fit_ssp,
    processing_contract,
)

EEG = ("F3", "F4", "C3", "C4", "P3", "P4")
CONTRACT = processing_contract(
    eeg_channels=EEG,
    eog_channels=("EOG",),
    out_sfreq=128.0,
    stamp="notch60/band1-45-o3/soxr-LQ",
)


def contract() -> dict:
    """A fresh, JSON-normalized copy of the test contract."""

    return processing_contract(
        eeg_channels=EEG,
        eog_channels=("EOG",),
        out_sfreq=128.0,
        stamp="notch60/band1-45-o3/soxr-LQ",
    )


def projector(count: int = len(EEG)) -> np.ndarray:
    """A valid projector that removes exactly one EEG direction."""

    vector = np.full(count, 1.0 / np.sqrt(count))
    return np.eye(count) - np.outer(vector, vector)


def correlated_epochs(
    rng: np.random.Generator,
    events: int = 9,
    samples: int = 256,
    eeg_count: int = len(EEG),
    eog_count: int = 1,
    noise: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """EEG epochs dominated by a shared direction driven by the EOG channel."""

    weight = np.linspace(0.4, 1.0, eeg_count)
    eeg = np.empty((events, samples, eeg_count))
    eog = np.empty((events, samples, eog_count))
    for event in range(events):
        movement = rng.normal(0.0, 12.0, (samples, eog_count))
        eeg[event] = movement @ weight[None, :] + rng.normal(
            0.0, noise, (samples, eeg_count)
        )
        eog[event] = movement
    return eeg, eog


class ContractTests(unittest.TestCase):
    def test_contract_is_json_normalized(self) -> None:
        first = processing_contract(
            eeg_channels=("F3", "C3"), eog_channels=("EOG",), out_sfreq=128.0
        )
        second = processing_contract(
            eeg_channels=["F3", "C3"], eog_channels=["EOG"], out_sfreq=128.0
        )
        self.assertEqual(first, second)
        with self.assertRaises(ValueError):
            processing_contract(
                eeg_channels=("F3", "C3"), out_sfreq=float("nan")
            )


class OperatorTests(unittest.TestCase):
    def test_projector_must_be_sane(self) -> None:
        with self.assertRaises(ValueError):
            SpatialOperator(np.eye(len(EEG)), contract(), {})  # full rank
        with self.assertRaises(ValueError):
            SpatialOperator(np.eye(3), contract(), {})  # wrong shape
        with self.assertRaises(ValueError):
            SpatialOperator(np.full((len(EEG), len(EEG)), np.nan), contract(), {})

    def test_apply_and_contract_validation(self) -> None:
        operator = SpatialOperator(projector(), contract(), {"note": "x"})
        data = np.random.default_rng(1).normal(size=(10, len(EEG)))
        corrected = operator.apply(data)
        self.assertEqual(corrected.shape, data.shape)
        with self.assertRaises(ValueError):
            operator.apply(np.zeros((10, len(EEG) - 1)))
        other = processing_contract(
            eeg_channels=tuple(reversed(EEG)),
            eog_channels=("EOG",),
            out_sfreq=128.0,
        )
        with self.assertRaises(ValueError):
            operator.validate(other)

    def test_apply_window_is_once_only_and_preserves_eog(self) -> None:
        operator = SpatialOperator(projector(), contract(), {"note": "x"})
        data = np.random.default_rng(2).normal(size=(64, len(EEG)))
        eog = np.random.default_rng(3).normal(size=(64, 1))
        times = np.arange(64) / 128.0
        window = EEGWindow(
            data, eog, times, valid=True, reasons=(), start_sample=64,
            channel_names=EEG,
        )
        self.assertIsNone(window.artifact_id)
        operator.apply_window(window)
        self.assertEqual(window.artifact_id, operator.artifact_id)
        self.assertTrue(np.allclose(window.eog, eog))  # EOG untouched
        self.assertTrue(window.valid)
        self.assertEqual(window.reasons, ())
        with self.assertRaises(ValueError):
            operator.apply_window(window)  # already corrected

    def test_apply_window_rejects_unknown_channel_order(self) -> None:
        operator = SpatialOperator(projector(), contract(), {})
        window = EEGWindow(
            np.zeros((64, len(EEG))), np.zeros((64, 1)), np.arange(64) / 128.0,
            valid=True, reasons=(), start_sample=0, channel_names=tuple(reversed(EEG)),
        )
        with self.assertRaises(ValueError):
            operator.apply_window(window)

    def test_save_and_load_roundtrip(self) -> None:
        operator = SpatialOperator(projector(), contract(), {"events": 9})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "eye_operator.npz"
            operator.save(path)
            loaded = SpatialOperator.load(path)
            # Never silently overwrite an existing operator file.
            with self.assertRaises(FileExistsError):
                operator.save(path)
        self.assertTrue(np.allclose(loaded.matrix, operator.matrix))
        self.assertEqual(loaded.contract, operator.contract)
        self.assertEqual(loaded.report, operator.report)
        self.assertEqual(loaded.artifact_id, operator.artifact_id)


class FitTests(unittest.TestCase):
    def test_fit_reduces_heldout_eog_coupling(self) -> None:
        rng = np.random.default_rng(42)
        eeg, eog = correlated_epochs(rng)
        operator = fit_ssp(eeg, eog, contract(), n_components=1)
        self.assertTrue(operator.report["heldout_coupling_ratio"] <= 0.5)
        self.assertEqual(operator.contract, contract())

    def test_fit_rejects_poor_calibration(self) -> None:
        rng = np.random.default_rng(7)
        eeg, eog = correlated_epochs(rng)
        # Too few events.
        with self.assertRaises(ValueError):
            fit_ssp(eeg[:4], eog[:4], contract())
        # Epochs too short.
        with self.assertRaises(ValueError):
            fit_ssp(eeg[:, :20], eog[:, :20], contract())
        # Non-finite data.
        bad = eeg.copy()
        bad[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            fit_ssp(bad, eog, contract())
        # Channel counts must match the contract.
        with self.assertRaises(ValueError):
            fit_ssp(eeg[:, :, :5], eog, contract())
        # Flat EOG proves nothing.
        flat_eog = np.full_like(eog, 0.5)
        with self.assertRaises(ValueError):
            fit_ssp(eeg, flat_eog, contract())
        # Unsupported component count.
        with self.assertRaises(ValueError):
            fit_ssp(eeg, eog, contract(), n_components=0)
        with self.assertRaises(ValueError):
            fit_ssp(eeg, eog, contract(), n_components=3)

    def test_fit_raises_when_coupling_is_not_reduced(self) -> None:
        # Independent EEG and EOG: nothing real to remove, so the held-out
        # coupling cannot drop enough.
        rng = np.random.default_rng(11)
        eeg = rng.normal(0.0, 8.0, (18, 256, len(EEG)))
        eog = rng.normal(0.0, 8.0, (18, 256, 1))
        with self.assertRaisesRegex(ValueError, "coupling"):
            fit_ssp(eeg, eog, contract())


class EpochCuttingTests(unittest.TestCase):
    """cut_epochs: the already-processed-data input boundary for C."""

    def setUp(self) -> None:
        # 6 s @ 100 Hz; every value encodes its own row for easy checks.
        self.sfreq = 100.0
        rows = 600
        self.times = np.arange(rows) / self.sfreq
        self.eeg = np.stack([np.arange(rows), np.arange(rows) * 2.0], axis=1)
        self.eog = np.arange(rows).reshape(-1, 1)

    def test_epochs_are_cut_around_event_centers(self) -> None:
        epochs, eog_epochs = cut_epochs(
            self.eeg, self.eog, self.times,
            event_times=(1.0, 3.5), seconds=1.0,
        )
        self.assertEqual(epochs.shape, (2, 100, 2))
        self.assertEqual(eog_epochs.shape, (2, 100, 1))
        # Center row 100 (t=1.0) sits in the middle of the first epoch.
        center = 50  # 100 rows, center index within the slice
        self.assertEqual(epochs[0, center, 0], 100.0)
        self.assertEqual(eog_epochs[0, center, 0], 100.0)

    def test_epochs_must_fit_inside_the_data(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside"):
            cut_epochs(
                self.eeg, self.eog, self.times,
                event_times=(0.01,), seconds=1.0,
            )
        with self.assertRaisesRegex(ValueError, "outside"):
            cut_epochs(
                self.eeg, self.eog, self.times,
                event_times=(5.9,), seconds=1.0,
            )

    def test_inputs_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            cut_epochs(self.eeg, self.eog[:100], self.times, (1.0,))
        with self.assertRaises(ValueError):
            cut_epochs(self.eeg, self.eog, self.times[:100], (1.0,))
        dup_times = self.times.copy()
        dup_times[10] = dup_times[9]  # not strictly increasing
        with self.assertRaises(ValueError):
            cut_epochs(self.eeg, self.eog, dup_times, (1.0,))
        bad_times = self.times.copy()
        bad_times[5] = np.nan
        with self.assertRaises(ValueError):
            cut_epochs(self.eeg, self.eog, bad_times, (1.0,))
        with self.assertRaises(ValueError):
            cut_epochs(self.eeg, self.eog, self.times, (3.0, 3.0))
        with self.assertRaises(ValueError):
            cut_epochs(self.eeg, self.eog, self.times, ())


if __name__ == "__main__":
    unittest.main()
