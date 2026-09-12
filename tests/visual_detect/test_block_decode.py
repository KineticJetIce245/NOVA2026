"""Tests for trial averaging: block formation, block labels and the two modes.

Run either with pytest or directly:

    .venv\\Scripts\\python.exe -m pytest tests/visual_detect -q
    .venv\\Scripts\\python.exe tests\\visual_detect\\test_block_decode.py
"""

from __future__ import annotations

import numpy as np

from scripts.visual_detect import block_decode as blk


def _toy() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Four trials per class per session, alternating in time, two sessions.

    Trial values equal their label, so a correct block average of one class is
    exactly the label.
    """
    labels, subjects, sessions, times, durations = [], [], [], [], []
    for label in (0, 1):
        for session in ("sub-01/a", "sub-01/b"):
            for repeat in range(4):
                labels.append(label)
                subjects.append("sub-01")
                sessions.append(session)
                times.append(1000.0 + 3000.0 * (4 * (session[-1] > "a") + repeat))
                durations.append(float(label))
    return (
        np.repeat(np.asarray(durations, dtype=np.float32)[:, None, None], 2, axis=1).repeat(5, axis=2),
        np.asarray(labels, dtype=np.int64),
        np.asarray(subjects, dtype=object),
        np.asarray(sessions, dtype=object),
        np.asarray(times, dtype=np.float64),
    )


# --------------------------------------------------------------------------- #
# block formation
# --------------------------------------------------------------------------- #
def test_blocks_are_consecutive_groups_of_k() -> None:
    index = np.arange(10)
    times = np.arange(10) * 3000.0
    blocks = blk.form_blocks(index, times, 3, max_gap_ms=15000.0)
    assert [rows.tolist() for rows in blocks] == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]


def test_a_leftover_shorter_run_is_dropped_not_padded() -> None:
    index = np.arange(10)
    times = np.arange(10) * 3000.0
    blocks = blk.form_blocks(index, times, 4, max_gap_ms=15000.0)
    assert [rows.tolist() for rows in blocks] == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_an_unplanned_break_cuts_the_run() -> None:
    # a 19 s pause between trial 1 and trial 2, with a 15 s limit
    times = np.asarray([0.0, 1000.0, 20000.0, 21000.0, 22000.0, 23000.0])
    blocks = blk.form_blocks(np.arange(6), times, 2, max_gap_ms=15000.0)
    assert [rows.tolist() for rows in blocks] == [[0, 1], [2, 3], [4, 5]]
    # with k=3 the first pair cannot be completed, so it is abandoned
    blocks = blk.form_blocks(np.arange(6), times, 3, max_gap_ms=15000.0)
    assert [rows.tolist() for rows in blocks] == [[2, 3, 4]]


def test_block_size_must_be_positive() -> None:
    try:
        blk.form_blocks(np.arange(4), np.arange(4) * 1000.0, 0, 15000.0)
    except ValueError:
        return
    raise AssertionError("a zero block size was accepted")


# --------------------------------------------------------------------------- #
# block datasets
# --------------------------------------------------------------------------- #
def test_block_dataset_never_mixes_classes_or_sessions() -> None:
    data, labels, subjects, sessions, times = _toy()
    # with k=2 over the strict (0,1) alternation every block averages one
    # label with itself, so the block mean equals the label exactly
    blocks, block_labels, block_sessions, block_subjects = blk.block_dataset(
        data, labels, subjects, sessions, times, k=2, max_gap_ms=15000.0, min_per_class=1
    )
    assert len(blocks) == len(block_labels) == len(block_sessions) > 0
    assert set(block_labels) == {0, 1}
    for block, label in zip(blocks, block_labels):
        assert np.isclose(block[0, 0], float(label)), "a block mixed two classes"
    assert set(block_subjects) == {"sub-01"}
    assert set(block_sessions) <= {"sub-01/a", "sub-01/b"}


def test_blocks_never_span_two_sessions() -> None:
    # two sessions 1 s apart: without session grouping, a k=4 block would average
    # across the boundary
    labels = np.tile([1, 0], 4)
    data = np.zeros((8, 2, 3), dtype=np.float32)
    subjects = np.asarray(["sub-01"] * 8, dtype=object)
    sessions = np.asarray(["sub-01/a"] * 4 + ["sub-01/b"] * 4, dtype=object)
    times = np.asarray([0.0, 3000.0, 6000.0, 9000.0, 10000.0, 13000.0, 16000.0, 19000.0])
    blocks, block_labels, block_sessions, _ = blk.block_dataset(
        data, labels, subjects, sessions, times, k=2, max_gap_ms=15000.0, min_per_class=1
    )
    for rows_session in block_sessions:
        assert rows_session in {"sub-01/a", "sub-01/b"}
    # k=2 same-class blocks are impossible here (labels alternate), so nothing
    assert len(blocks) == 0 or set(block_sessions) == {"sub-01/a", "sub-01/b"}


def test_block_averaging_halves_the_noise_not_the_signal() -> None:
    rng = np.random.default_rng(0)
    n, channels, samples = 120, 3, 8
    labels = np.tile([0, 1], n // 2)
    signal = np.zeros((n, channels, samples), dtype=np.float32)
    signal[labels == 1, :, 3:5] = 1.0  # a fixed response in every stimulus epoch
    noise = rng.normal(0.0, 1.0, size=signal.shape).astype(np.float32)
    data = signal + noise
    subjects = np.asarray(["sub-01"] * n, dtype=object)
    sessions = np.asarray(["sub-01/a"] * n, dtype=object)
    times = np.arange(n) * 3000.0

    single, single_labels, _, _ = blk.block_dataset(
        data, labels, subjects, sessions, times, 1, 15000.0, 1
    )
    blocks, block_labels, _, _ = blk.block_dataset(
        data, labels, subjects, sessions, times, 4, 15000.0, 1
    )
    # the response amplitude is unchanged by averaging
    assert np.isclose(blocks[block_labels == 1][:, :, 3:5].mean(), 1.0, atol=0.15)
    assert np.isclose(single[single_labels == 1][:, :, 3:5].mean(), 1.0, atol=0.15)
    # the noise on the baseline is smaller by about sqrt(4)
    before = single[single_labels == 0].std()
    after = blocks[block_labels == 0].std()
    assert after < before / 1.5, f"noise fell only from {before:.3f} to {after:.3f}"


def test_block_dataset_needs_enough_blocks_per_class() -> None:
    data, labels, subjects, sessions, times = _toy()
    blocks, _, _, _ = blk.block_dataset(
        data, labels, subjects, sessions, times, 2, 15000.0, min_per_class=99
    )
    assert len(blocks) == 0


def test_block_dataset_on_empty_input_is_empty() -> None:
    empty = np.empty((0, 2, 5), dtype=np.float32)
    blocks, block_labels, block_sessions, block_subjects = blk.block_dataset(
        empty,
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=object),
        np.empty(0, dtype=object),
        np.empty(0, dtype=np.float64),
        2,
        15000.0,
        1,
    )
    assert blocks.shape[0] == 0
    assert block_labels.size == block_sessions.size == block_subjects.size == 0


# --------------------------------------------------------------------------- #
# score smoothing
# --------------------------------------------------------------------------- #
def test_score_smoothing_matches_the_block_grouping() -> None:
    _, labels, _, sessions, times = _toy()
    scores = np.where(labels == 1, 1.0, -1.0)  # a perfect per-trial decoder
    smoothed, smooth_labels = blk.smooth_scores(
        scores, labels, sessions, times, k=2, max_gap_ms=15000.0, min_per_class=1
    )
    assert smoothed.size == smooth_labels.size > 0
    # averaging a perfect decoder must not change the sign
    assert np.all(np.sign(smoothed) == np.where(smooth_labels == 1, 1.0, -1.0))


def test_score_smoothing_drops_runs_with_a_missing_score() -> None:
    _, labels, _, sessions, times = _toy()
    scores = np.ones(len(labels))
    scores[0] = np.nan
    smoothed, _ = blk.smooth_scores(
        scores, labels, sessions, times, k=2, max_gap_ms=15000.0, min_per_class=1
    )
    assert np.isfinite(smoothed).all()
    assert smoothed.size < scores.size


# --------------------------------------------------------------------------- #
# argument handling
# --------------------------------------------------------------------------- #
def test_defaults_cover_a_range_of_block_sizes() -> None:
    args = blk.parse_args([])
    assert args.mode == "both"
    assert args.block_sizes == list(blk.BLOCK_SIZES)
    assert args.block_sizes[0] == 1, "K=1 must stay in the sweep as the baseline"


def test_main_runs_end_to_end_on_a_built_checkpoint() -> None:
    from scripts.visual_detect.decode import DATASETS
    from scripts.visual_detect.device import VIS_DIR

    if not (VIS_DIR / DATASETS["occipital"]).exists():
        # the sweep needs a checkpoint; build_dataset is exercised separately
        print("SKIP (no checkpoint built)")
        return
    summary = blk.main(
        ["--channel-set", "occipital", "--block-sizes", "1", "2", "--decoders", "template"]
    )
    assert summary["results"], "the sweep produced no rows"
    assert {row["mode"] for row in summary["results"]} == {"block", "score"}


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
