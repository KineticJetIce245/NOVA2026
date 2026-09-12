"""Tests for the visual-detection helpers that carry the experiment's logic.

Run either with pytest or directly:

    .venv\\Scripts\\python.exe -m pytest tests/visual_detect -q
    .venv\\Scripts\\python.exe tests\\visual_detect\\test_visual_detect.py
"""

from __future__ import annotations

import numpy as np
import torch
from mne import Annotations, create_info
from mne.io import RawArray

from nova2026.config import DATA_DIR  # noqa: F401  (import path check)
from scripts.visual_detect import build_dataset as bd
from scripts.visual_detect import device as dev
from scripts.visual_detect import train_occipital as tr
from scripts.visual_detect.models import EEGNetWindow


# --------------------------------------------------------------------------- #
# device
# --------------------------------------------------------------------------- #
def test_channel_sets_are_cap_ordered_and_known() -> None:
    occipital = dev.channels_for("occipital")
    posterior = dev.channels_for("posterior")
    assert occipital == ["O1", "Oz", "O2", "PO7", "PO3", "POz", "PO4", "PO8"]
    # both must be ordered the way the recording lays channels out
    positions = [list(dev.EEG_CHANNELS).index(ch) for ch in posterior]
    assert positions == sorted(positions)
    assert set(occipital) <= set(posterior)


def test_unknown_channel_set_is_rejected() -> None:
    try:
        dev.channels_for("motor")
    except ValueError as exc:
        assert "motor" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unknown channel set was accepted")


def test_stem_is_filesystem_safe() -> None:
    assert dev.stem("occipital", -50, 500) == "occipital_m50_500ms"
    assert dev.stem("posterior", 100, 300) == "posterior_100_300ms"


# --------------------------------------------------------------------------- #
# trial structure
# --------------------------------------------------------------------------- #
def _raw(annotations: list[tuple[float, str]], seconds: float = 60.0) -> RawArray:
    data = np.zeros((2, int(seconds * 500.0)), dtype=np.float64)
    info = create_info(["O1", "Oz"], 500.0, "eeg")
    raw = RawArray(data, info, verbose=False)
    raw.set_annotations(
        Annotations(
            onset=[o for o, _ in annotations],
            duration=[0.0] * len(annotations),
            description=[d for _, d in annotations],
        )
    )
    return raw


def test_read_trials_pairs_stimulus_with_next_response() -> None:
    raw = _raw([(10.0, "13"), (10.3, "14"), (20.0, "13"), (20.75, "14")])
    trials = bd.read_trials(raw)
    assert trials.stim_ms.tolist() == [10000, 20000]
    assert np.allclose(trials.rt_ms, [300.0, 750.0])


def test_read_trials_drops_unanswered_and_premature_trials() -> None:
    raw = _raw(
        [
            (10.0, "13"),
            (10.9, "14"),
            (14.0, "12"),  # premature: fires before the 20 s stimulus
            (20.0, "13"),
            (20.75, "14"),
            (30.0, "13"),  # never answered
        ]
    )
    trials = bd.read_trials(raw)
    assert trials.stim_ms.tolist() == [10000, 20000, 30000]
    # the premature event sits between two responses, and the log does not say
    # which trial either press belongs to, so both are dropped
    assert np.isnan(trials.rt_ms[0]), "the trial before the premature event is ambiguous"
    assert np.isnan(trials.rt_ms[1]), "the trial after the premature event is ambiguous"
    assert np.isnan(trials.rt_ms[2]), "an unanswered trial has no RT"


def test_read_trials_keeps_a_trial_with_no_premature_press() -> None:
    raw = _raw([(10.0, "13"), (10.3, "14"), (12.0, "13"), (12.4, "14")])
    trials = bd.read_trials(raw)
    assert np.allclose(trials.rt_ms, [300.0, 400.0], atol=1e-6)
    assert trials.premature_ms.size == 0


# --------------------------------------------------------------------------- #
# null-window sampling
# --------------------------------------------------------------------------- #
def test_noise_regions_apply_the_guard_on_both_sides() -> None:
    regions = bd.noise_regions(10_000, [2000, 3000], 700)
    assert regions == [(0, 1300), (3700, 10_000)]


def test_noise_regions_keep_the_recording_edges() -> None:
    # two events 0.2 s apart with a 0.3 s guard block 0.1-0.9 s; the head and
    # tail of the recording stay usable, only the event vicinity is removed
    assert bd.noise_regions(1000, [400, 600], 300) == [(0, 100), (900, 1000)]


def test_noise_regions_of_a_real_session_are_long_enough() -> None:
    # sub-01/ses-S1: 90 stimuli 2.4-9.6 s apart, nothing else, 612 s long
    rng = np.random.default_rng(0)
    gaps = rng.uniform(2.4, 9.6, size=90)
    onsets = np.cumsum(gaps) * 1000.0
    regions = bd.noise_regions(612_000, onsets, 700)
    assert regions, "a PVT session must leave usable silence"
    assert max(e - s for s, e in regions) >= 500  # at least one epoch window


def test_sample_nulls_stay_inside_the_regions() -> None:
    regions = [(0, 1300), (3700, 10_000)]
    rng = np.random.default_rng(0)
    starts = bd.sample_nulls(regions, 5, window_ms=550, rng=rng)
    assert starts.size == 5
    for start in starts:
        assert any(lo <= start and start + 550 <= hi for lo, hi in regions)


def test_sample_nulls_never_exceeds_the_pool() -> None:
    starts = bd.sample_nulls([(0, 1100)], 50, window_ms=550, rng=np.random.default_rng(0))
    assert starts.tolist() == [0, 550]


def test_deterministic_seed_is_stable_across_processes() -> None:
    # hash() is salted per process for strings; this must not be
    assert bd.deterministic_seed("sub-01", "ses-S2") == bd.deterministic_seed(
        "sub-01", "ses-S2"
    )
    assert bd.deterministic_seed("sub-01", "ses-S2") != bd.deterministic_seed(
        "sub-02", "ses-S2"
    )


# --------------------------------------------------------------------------- #
# epoch geometry
# --------------------------------------------------------------------------- #
def test_windows_slices_and_baseline_corrects() -> None:
    # sample 0 is pre_ms = -50 ms at 128 Hz, so 100 ms is sample 19
    data = np.zeros((2, 8, 70), dtype=np.float32)
    data[:, :, :6] = 5.0  # baseline region (-50 .. 0 ms)
    data[:, :, 19:45] = 3.0  # the 100-300 ms window
    out = tr.windows(data, 128.0, -50, 100, 300, (-50, 0), zscore=False)
    assert out.shape == (2, 8, 26)
    assert np.allclose(out, 3.0 - 5.0)


def test_windows_scaling_keeps_the_window_mean() -> None:
    # the evoked response lives in the window mean, so scaling must not centre
    rng = np.random.default_rng(0)
    data = rng.normal(5.0, 3.0, size=(4, 8, 70)).astype(np.float32)
    data[:, :, 19:45] += 2.0  # a stimulus-like level shift inside the window
    raw = tr.windows(data, 128.0, -50, 100, 300, (-50, 0), zscore=False)
    scaled = tr.windows(data, 128.0, -50, 100, 300, (-50, 0), zscore=True)
    assert np.allclose(scaled.std(axis=(1, 2)), 1.0, atol=1e-5)
    # scaling is positive, so the sign of the shift survives
    assert np.all(np.sign(scaled.mean(axis=(1, 2))) == np.sign(raw.mean(axis=(1, 2))))
    assert np.abs(scaled.mean(axis=(1, 2))).min() > 0.0


def test_windows_rejects_impossible_geometry() -> None:
    data = np.zeros((1, 8, 70), dtype=np.float32)
    for bad in ((600, 800), (200, 100), (-200, -100)):
        try:
            tr.windows(data, 128.0, -50, bad[0], bad[1], (-50, 0), zscore=False)
        except ValueError:
            continue
        raise AssertionError(f"window {bad} was accepted")


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_macro_f1_is_zero_for_a_single_class() -> None:
    assert tr.macro_f1(np.ones(10), np.ones(10)) == 0.0


def test_mean_auc_averages_over_groups() -> None:
    truth = np.array([0, 0, 1, 1, 0, 1])
    scores = np.array([-1.0, -2.0, 1.0, 2.0, -1.0, 1.0])
    groups = np.array(["a", "a", "a", "a", "b", "b"])
    assert np.isclose(tr.mean_auc(truth, scores, groups), 1.0)


def test_subject_column_accepts_rows_and_tables() -> None:
    rows = np.empty(3, dtype=object)
    rows[:] = [(s,) for s in ("sub-01", "sub-02", "sub-01")]
    assert tr.subject_column(rows).tolist() == ["sub-01", "sub-02", "sub-01"]
    table = np.array([["sub-01"], ["sub-02"]], dtype=object)
    assert tr.subject_column(table).tolist() == ["sub-01", "sub-02"]


def test_permutation_null_is_chance_level() -> None:
    rng = np.random.default_rng(0)
    truth = rng.integers(0, 2, size=400)
    scores = rng.normal(size=400)
    null = tr.permutation_null(truth, scores, n_perm=50, seed=0)
    assert 0.4 < null["accuracy_mean"] < 0.6
    assert 0.4 < null["macro_f1_mean"] < 0.6


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def test_eegnet_window_accepts_the_default_geometry() -> None:
    model = EEGNetWindow(chn=8, fs=128.0, t=26, kernel_ms=125.0, pool1=2, pool2=4)
    assert model.kernel == 16
    # block 1 halves 26 -> 13, block 2 quarters 13 -> 3, times f1 * d
    assert model.features == 8 * 2 * 3
    out = model(torch.zeros(4, 8, 26))
    assert out.shape == (4, 2)


def test_eegnet_window_skips_the_second_pool_when_asked() -> None:
    model = EEGNetWindow(chn=8, fs=128.0, t=26, pool1=2, pool2=None)
    assert model.features == 8 * 2 * 13
    assert model(torch.zeros(2, 8, 26)).shape == (2, 2)


def test_eegnet_window_rejects_a_collapsed_window() -> None:
    try:
        EEGNetWindow(chn=8, fs=128.0, t=26, pool1=4, pool2=8)
    except ValueError as exc:
        assert "erase" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("an over-pooled window was accepted")


def test_eegnet_window_checks_the_input_shape() -> None:
    model = EEGNetWindow(chn=8, fs=128.0, t=26)
    try:
        model(torch.zeros(2, 8, 30))
    except ValueError as exc:
        assert "(8, 26)" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("wrongly shaped input was accepted")


if __name__ == "__main__":
    import traceback

    failures = 0
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            try:
                func()
                print(f"PASS {name}")
            except Exception:
                failures += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    print(f"\n{len([n for n in globals() if n.startswith('test_')]) - failures} passed, "
          f"{failures} failed")
    raise SystemExit(1 if failures else 0)
