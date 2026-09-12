"""Tests for the visual-detection pipeline and the waveform-keeping decoders.

Run either with pytest or directly:

    .venv\\Scripts\\python.exe -m pytest tests/visual_detect -q
    .venv\\Scripts\\python.exe tests\\visual_detect\\test_decode.py
"""

from __future__ import annotations

import numpy as np
from mne import create_info
from mne.io import RawArray

from scripts.visual_detect import decode as dec
from scripts.visual_detect import device as dev
from scripts.visual_detect import pipeline as pipe


def _raw(values: np.ndarray, sfreq: float = 500.0) -> RawArray:
    info = create_info(list(pipe.EEG_CHANNELS), sfreq, "eeg")
    return RawArray(values.astype(np.float64), info, verbose=False)


def _planted(
    n: int = 100, channels: int = 8, samples: int = 26, amplitude: float = 1.2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Balanced epochs with a short positive bump in the first channel."""
    rng = np.random.default_rng(seed)
    labels = np.tile([1, 0], n // 2)
    data = rng.normal(0.0, 1.0, size=(n, channels, samples)).astype(np.float32)
    data[labels == 1, 0, samples // 3 : samples // 3 + 4] += amplitude
    return data, labels


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #
def test_pipeline_rejects_an_unreachable_band() -> None:
    for kwargs in (
        {"h_freq": 80.0, "sample_rate": 128.0},  # above the resampled Nyquist
        {"h_freq": 260.0, "sample_rate": 500.0},  # above the raw Nyquist
        {"l_freq": 30.0, "h_freq": 20.0},  # inverted
        {"l_freq": 0.0},  # no high-pass
    ):
        try:
            pipe.VisualPipeline(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"pipeline accepted {kwargs}")


def test_pipeline_band_and_rate_reach_the_recording() -> None:
    raw = _raw(np.random.default_rng(0).normal(0, 1, size=(62, 3000)))
    pipe.VisualPipeline(l_freq=4.0, h_freq=20.0, sample_rate=128.0).run([raw])
    assert np.isclose(raw.info["sfreq"], 128.0)
    assert raw.n_times == 768


def test_pipeline_without_normalisation_keeps_microvolts() -> None:
    # 10 uV of a 6 Hz sine survives the 4-20 Hz band-pass
    time = np.arange(3000) / 500.0
    signal = 10e-6 * np.sin(2 * np.pi * 6.0 * time)
    values = np.tile(signal, (62, 1))
    raw = _raw(values)
    pipe.VisualPipeline(normalize=False, sample_rate=128.0).run([raw])
    peak_uv = np.abs(raw.get_data() * 1e6).max()
    assert 8.0 < peak_uv < 12.0, f"expected ~10 uV, got {peak_uv}"


def test_pipeline_normalisation_shrinks_an_evoked_response() -> None:
    time = np.arange(3000) / 500.0
    signal = 10e-6 * np.sin(2 * np.pi * 6.0 * time)
    values = np.tile(signal, (62, 1))
    raw = _raw(values)
    pipe.VisualPipeline(normalize=True, sample_rate=128.0).run([raw])
    peak_uv = np.abs(raw.get_data() * 1e6).max()
    # centre/scale/clip divides by 8 uV, so the 10 uV wave drops well below it
    assert peak_uv < 5.0, f"expected the response to shrink, got {peak_uv}"


def test_pipeline_rejects_a_non_raw() -> None:
    try:
        pipe.VisualPipeline().run([np.zeros((2, 10))])
    except TypeError:
        return
    raise AssertionError("a non-Raw input was accepted")


def test_missing_channels_reports_what_is_absent() -> None:
    assert pipe.missing_channels(["O1", "Fz"], ["O1"]) == ["Fz"]
    assert pipe.missing_channels(["O1"], ["O1", "Fz"]) == []


# --------------------------------------------------------------------------- #
# channel sets (the 8 -> 17 widening)
# --------------------------------------------------------------------------- #
def test_the_seventeen_electrode_set_contains_the_eight() -> None:
    occipital = dev.channels_for("occipital")
    posterior = dev.channels_for("posterior")
    assert len(occipital) == 8
    assert len(posterior) == 17
    assert set(occipital) <= set(posterior)
    assert set(posterior) - set(occipital) == {
        "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "Pz",
    }


def test_the_full_set_is_the_whole_cap() -> None:
    assert dev.channels_for("all") == list(dev.EEG_CHANNELS)
    assert len(dev.channels_for("all")) == 62


# --------------------------------------------------------------------------- #
# decoders
# --------------------------------------------------------------------------- #
def _score(decoder: str, data, labels, args=None) -> float:
    from sklearn.metrics import roc_auc_score

    args = args or dec.parse_args([])
    train = np.arange(0, 60)
    test = np.arange(60, len(data))
    result = dec.DECODERS[decoder](data[train], labels[train], data[test], args)
    return float(roc_auc_score(labels[test], result.values))


def test_every_decoder_beats_chance_on_a_planted_response() -> None:
    # 2.5 uV over a 1 uV background: strong enough that a covariance-based
    # decoder, which needs more evidence than a template, also finds it
    data, labels = _planted(amplitude=2.5)
    for name in dec.DECODERS:
        auc = _score(name, data, labels)
        assert auc > 0.6, f"{name} did not find a planted response (AUC={auc:.3f})"


def test_decoders_stay_near_chance_without_a_response() -> None:
    rng = np.random.default_rng(1)
    data = rng.normal(0.0, 1.0, size=(100, 8, 26)).astype(np.float32)
    labels = np.tile([1, 0], 50)
    for name in ("template", "xdawnd", "linear"):
        auc = _score(name, data, labels)
        assert 0.4 < auc < 0.6, f"{name} invented a response (AUC={auc:.3f})"


def test_window_mean_ignores_the_waveform() -> None:
    # a bump that leaves the window mean untouched must not help
    data, labels = _planted(amplitude=1.0)
    data[labels == 1, 0, 8] -= 1.0  # add a negative mirror of the bump
    auc = _score("window_mean", data, labels)
    assert 0.4 < auc < 0.6, f"window_mean read a zero-mean bump (AUC={auc:.3f})"


def test_templates_are_unit_norm_and_separated() -> None:
    data, labels = _planted()
    templates = dec.fit_templates(data, labels)
    assert np.isclose(np.linalg.norm(templates[0]), 1.0)
    assert np.isclose(np.linalg.norm(templates[1]), 1.0)
    assert not np.allclose(templates[0], templates[1])


def test_template_scores_need_no_fitting() -> None:
    data, labels = _planted()
    templates = dec.fit_templates(data, labels)
    first = dec.template_scores(data, templates)
    second = dec.template_scores(data, templates)
    assert np.allclose(first, second), "template scoring must be deterministic"


def test_xdawn_filters_have_one_row_per_class_and_filter() -> None:
    data, labels = _planted()
    filters = dec.xdawn_filters(data, labels, n_filters=3)
    assert filters.shape == (6, data.shape[1])
    projected = dec.apply_filters(data, filters)
    assert projected.shape == (len(data), 6, data.shape[2])


def test_xdawn_needs_both_classes() -> None:
    data, _ = _planted()
    try:
        dec.xdawn_filters(data, np.ones(len(data), dtype=int), 2)
    except ValueError:
        return
    raise AssertionError("xDAWN accepted a single-class training set")


def test_riemannian_features_are_a_true_embedding() -> None:
    data, labels = _planted(channels=4)
    features, reference = dec.riemannian_tangent(data)
    assert features.shape == (len(data), 4 * 5 // 2)
    assert reference.shape == (4, 4)
    again, _ = dec.riemannian_tangent(data, reference=reference)
    # projecting onto a supplied reference must be deterministic and finite
    assert np.allclose(features, again)
    assert np.isfinite(features).all()


def test_riemannian_reference_comes_from_the_training_trials() -> None:
    data, _ = _planted(channels=4)
    train_features, reference = dec.riemannian_tangent(data[:50])
    test_features, _ = dec.riemannian_tangent(data[50:], reference=reference)
    assert test_features.shape[0] == 50
    # the reference must be the geometric mean of the *training* trials, so a
    # test fold projected onto it keeps finite, non-degenerate features
    assert np.isfinite(test_features).all()
    assert test_features.std() > 0.0
    # re-projecting the training set onto its own reference must not change it
    again, _ = dec.riemannian_tangent(data[:50], reference=reference)
    assert np.allclose(train_features, again)


def test_riemannian_needs_more_than_one_sample_per_trial() -> None:
    data, _ = _planted(channels=4, samples=1)
    try:
        dec.riemannian_tangent(data)
    except ValueError as exc:
        assert "2 samples" in str(exc)
    else:
        raise AssertionError("a single-sample trial was accepted")


def test_parse_window_rejects_an_inverted_range() -> None:
    assert dec.parse_window("50-500") == (50, 500)
    try:
        dec.parse_window("300-100")
    except ValueError:
        return
    raise AssertionError("an inverted window was accepted")


def test_main_rejects_an_unknown_decoder() -> None:
    try:
        dec.main(["--decoders", "nope"])
    except SystemExit:
        return
    raise AssertionError("an unknown decoder was accepted")


if __name__ == "__main__":
    import traceback

    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            failures += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures} passed, {failures} failed")
    raise SystemExit(1 if failures else 0)
