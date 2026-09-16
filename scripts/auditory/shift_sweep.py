"""Step 5.5: what a timing offset between audio and EEG costs the decoder.

Every alignment claim in this project -- the live path's ``audio_start_lsl``
mapping, the ``<= 0.75 s`` gain-activation window, the window length, and
``MIN_MARGIN`` -- rests on one unmeasured assumption: that a small
audio/EEG misalignment costs little decoding accuracy. This module turns that
assumption into a curve. It sweeps the reference envelope against a fixed EEG
stream over a grid of offsets and reports how the decoder's balanced accuracy
decays away from the peak.

The offset swept here is the same quantity ``scripts/auditory/runner.py``
expresses as ``audio_offset``: **audio sample zero sits at trial start plus
``audio_offset``**. A positive offset moves the audio later in real time, so the
reference amplitude the decoder should compare against EEG sample ``t`` is the
envelope's value at ``t - offset``. ``shifted_envelope`` applies exactly that,
and ``tests/test_shift_sweep.py`` pins the direction by comparing against the
full ``replay_windows`` chain, which carries the same parameter.

Two design choices are worth stating, because neither is forced:

1. **The sweep shifts the cached envelope, not the signal chain.** The feature
   cache under ``datasets/auditory_features/<key>/`` already holds the two
   reference envelopes sampled on the chain's own 64 Hz grid, for both channel
   contracts. A shift of the envelope against a fixed EEG stream is
   mathematically what ``audio_offset`` does to the chain's output (the EEG
   stream does not depend on the audio at all), and it costs one cache read per
   trial instead of one full chain re-run per trial per offset. The direction is
   not assumed: it is tested against ``replay_windows``.

2. **The models are fitted at zero offset, once per fold, and then evaluated at
   every offset.** Refitting at each offset would answer a different question
   ("can a decoder re-learn a fixed misalignment?") and would multiply the cost
   by the size of the grid. The question here is what a *deployed* model loses
   when the streams are misaligned, which is the zero-offset model's answer.

The grid is in seconds and every point is a multiple of 12.5 ms, which is an
exact number of 80 Hz samples but only 0.8 of a 64 Hz sample: the pairing is
therefore linear interpolation on the envelope's own grid rather than a round to
the nearest sample, so the swept offset is the offset that was asked for.

**The pairing is measured, not derived.** ``--audit`` runs the raw chain at a
handful of offsets and compares its ``aligned.envelopes`` against the cached
envelope at the sweep's own position formula, window by window, and against the
mirror image of that formula. The first attempt at this experiment got the sign
backwards -- it read the envelope *later* for a positive offset -- and lost
nothing but truncation at zero offset, so the at-zero reproduction passed while
every other point was the mirror image of the truth. The audit's two numbers
(``envelope_max_difference`` and ``envelope_mirror_max_difference``) are what
pins the direction; the second is expected to be four orders of magnitude larger.

    python -B -m scripts.auditory.shift_sweep --out results --audit
"""

import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from . import aad_ridge, feature_cache, train_kuleuven
from .kuleuven_contract import REPO, RESULTS
from .outputs import guard_outputs

#: The frozen feature cache this experiment was run against. A different key
#: means a different feature contract, and the numbers below stop being step 5's.
EXPECTED_CACHE_KEY = "22e5546fa44abe0f"
MAIN_HISTORY = train_kuleuven.MAIN_HISTORY
ALPHA = train_kuleuven.DEFAULT_ALPHA
CONTRACTS = ("64ch", "20ch")
STEP_SECONDS = 0.0125
HALF_RANGE_SECONDS = 0.3
SUBSTEPS_PER_STEP = 2

#: Step 5's published pooled held-out-story numbers at zero offset, exactly as
#: ``results/aad_20260913-022712.json`` prints them. The sweep refuses to run
#: when its own zero point does not reproduce these, and
#: ``scripts/auditory/tests/test_shift_sweep.py`` asserts them to 1e-9: a harness
#: that disagreed with step 5 would be measuring itself, not the offset.
REFERENCE_BALANCED = {"64ch": 0.6194721652821231, "20ch": 0.5974097124824684}
REFERENCE_ACCURACY = {"64ch": 0.6170154185022027, "20ch": 0.5940941629955947}
REFERENCE_WINDOWS = 14528
#: The CLI's own guard, rounded to what a report quotes, so a rerun on a
#: different BLAS is not stopped over 1e-15 of summation order.
REFERENCE_TOLERANCE = 1e-3
CHANCE_BALANCED = 0.5


def times_grid(half_range=HALF_RANGE_SECONDS, step=STEP_SECONDS,
               substeps=SUBSTEPS_PER_STEP):
    """The offset grid: ``step`` over the range, ``substeps`` extra points near zero.

    The fine points are the ones that can be reported as multiples of the step,
    so the whole grid stays on one lattice and a reader never has to wonder
    which resolution a quoted peak was found at.
    """

    outer = np.arange(-half_range, half_range + step / 2, step)
    substep = step / substeps
    fine = substep * np.arange(1, substeps)
    grid = np.unique(np.concatenate([outer, fine, -fine]))
    return np.round(grid, 9)


def shift_for(offset, rate):
    """Envelope samples read *earlier* for an audio offset in seconds.

    Positive when the audio starts later, which is what makes the offset
    positive: the reference the EEG at a given moment is compared against is the
    audio that reached the ear ``shift`` samples ago. Kept as whole samples for
    bookkeeping and reporting; the pairing itself is fractional
    (:func:`positions_for`), because 12.5 ms is 0.8 of a 64 Hz sample.
    """

    return int(round(float(offset) * rate))


def positions_for(start, count, offset, rate):
    """Fractional envelope positions paired with a window's ``count`` rows.

    Row ``i`` of the design matrix predicts the EEG at the window's position
    ``start + i`` (its first ``lag_samples`` rows are dropped by the lag
    expansion, not shifted). The chain puts audio sample zero at trial start
    plus ``offset``, so the amplitude that row must be compared against is the
    envelope at ``start + i - offset * rate``.

    This formula is not derived from the chain's source and left to trust: it is
    measured against the chain window by window in
    :func:`compare_with_chain`, which reports ``max |chain - cached|`` for this
    position and for its mirror image.
    """

    return start + np.arange(count, dtype=float) - float(offset) * rate


def paired_envelope(envelope, positions, grid=None):
    """Envelope sampled at fractional positions, or ``None`` if any is outside.

    Linear interpolation on the envelope's own 64 Hz grid, which is what
    :meth:`EnvelopeBuffer.align` does with ``np.interp`` against the buffer's
    own timestamps. ``grid`` can be passed in because this is called once per
    window per offset, and rebuilding a 25 000-point axis there dominates the
    sweep's runtime.

    ``None`` means the chain would have refused this window: it marks a window
    ``audio_unavailable`` whenever its span is not covered by the audio received
    so far. Clamping, padding or scoring a shortened row set instead would
    invent a window the chain does not have, and it would also compare the
    reconstruction against an envelope that was centered over different rows
    than the ones it is correlated with.
    """

    if len(positions) == 0:
        return None
    if np.any(positions < 0.0) or np.any(positions > len(envelope) - 1):
        return None
    if grid is None:
        grid = np.arange(len(envelope), dtype=float)
    values = np.empty((len(positions), envelope.shape[1]), dtype=float)
    for candidate in range(envelope.shape[1]):
        values[:, candidate] = np.interp(positions, grid, envelope[:, candidate])
    return values


def trial_scores(model, signal, envelopes, start, length, offsets, rate):
    """Correlations of one window against both candidates, per offset.

    This is :meth:`RidgeDecoder.score` for one window and many offsets, with the
    design matrix hoisted out of the offset loop: it does not depend on the
    envelope, so it is built once. The arithmetic inside the loop is otherwise
    identical -- the reconstruction is centered over all ``count`` rows and the
    envelope over the same rows -- and ``tests/test_shift_sweep.py`` asserts
    equality with ``model.score`` on real cached windows.

    Rows that a given offset pushes outside the recording yield ``nan`` for that
    whole window, which :func:`sweep` reads as the chain's ``audio_unavailable``
    and drops rather than scoring a shortened window.
    """

    design = np.asarray(signal[start : start + length], dtype=float)
    normalized = (design - model.mean) / model.scale
    count = length - model.config.lag_samples
    rows = np.concatenate([normalized[lag : lag + count]
                           for lag in range(model.config.lag_samples + 1)], axis=1)
    reconstruction = rows @ model.weights
    reconstruction = reconstruction - reconstruction.mean()
    norm = np.linalg.norm(reconstruction)
    scores = np.full((len(offsets), 2), np.nan)
    grid = np.arange(len(envelopes), dtype=float)
    for index, offset in enumerate(offsets):
        if float(offset) == 0.0:
            # Step 5's own feature, as a plain slice: the zero point of the
            # sweep must not be an interpolation of itself.
            target = envelopes[start : start + count]
        else:
            target = paired_envelope(
                envelopes, positions_for(start, count, offset, rate), grid
            )
            if target is None:
                continue
        target = target - target.mean(axis=0)
        denominator = norm * np.linalg.norm(target, axis=0)
        columns = np.flatnonzero(denominator > 1e-12)
        if len(columns) == 0:
            continue
        scores[index, columns] = (
            reconstruction @ target[:, columns]
        ) / denominator[columns]
    return scores


def fold_models(corpus, contract, history, alpha=ALPHA, progress=None):
    """One model per held-out story fold, fitted at zero offset only.

    ``held_out_story_folds`` runs a guard that the validation side carries both
    classes; that guard is part of step 5's protocol and is reused here rather
    than reimplemented.
    """

    folds = train_kuleuven.held_out_story_folds(corpus)
    wanted = {trial.key for fold in folds for trial in fold["validation"]}
    subtotals = {trial.key: None for trial in corpus.trials if trial.key in wanted}
    total = None
    for trial in corpus.trials:
        features = feature_cache.load(trial.path, corpus.key)
        length = round(history * feature_cache.sample_rate())
        starts = [start for start, _ in features.windows(contract, history, hop=history)]
        moments = aad_ridge.trial_moments(
            features.signal(contract).astype(float),
            features.envelopes.astype(float),
            features.label, starts, length, features.metadata["lag_samples"],
        )
        if moments is None:
            raise ValueError(f"{trial.path} contributed no labeled window.")
        total = moments if total is None else total + moments
        if trial.key in subtotals:
            held = subtotals[trial.key]
            subtotals[trial.key] = moments if held is None else held + moments
        if progress is not None:
            progress(trial)
    models = []
    for fold in folds:
        training = total
        for trial in fold["validation"]:
            training = training - subtotals[trial.key]
        models.append({
            "fold": fold["name"],
            "validation": fold["validation"],
            "model": aad_ridge.fit(
                training, corpus.contract_of(contract, history), alpha,
                training_info={
                    "training": [(t.subject, t.trial_id, t.group) for t in fold["training"]],
                    "held_out": [(t.subject, t.trial_id, t.group) for t in fold["validation"]],
                    "scheme": "held_out_story", "fold": fold["name"],
                    "contract": contract, "history_seconds": history, "alpha": alpha,
                    "cache_key": corpus.key, "labels_in_inference_path": False,
                    "audio_offset_seconds": 0.0,
                },
            ),
        })
    return models


def sweep(corpus, contract, history, offsets, progress=None):
    """Score every held-out window at every offset; return decisions and truths.

    One cache read per trial serves the whole grid, and the fold a trial belongs
    to decides the model that scores it, so the protocol is step 5's
    held-out-story pooling at every offset.
    """

    rate = feature_cache.sample_rate()
    length = round(history * rate)
    models = fold_models(corpus, contract, history)
    by_trial = {}
    for record in models:
        for trial in record["validation"]:
            by_trial.setdefault(trial.key, []).append(record["model"])
    collected_decisions = [[] for _ in offsets]
    collected_truths = [[] for _ in offsets]
    for trial in corpus.trials:
        owners = by_trial.get(trial.key)
        if not owners:
            continue
        features = feature_cache.load(trial.path, corpus.key)
        signal = features.signal(contract).astype(float)
        for start, _ in features.windows(contract, history, hop=history):
            scores = trial_scores(owners[0], signal, features.envelopes, start,
                                  length, offsets, rate)
            for index in range(len(offsets)):
                first, second = scores[index]
                if not (np.isfinite(first) and np.isfinite(second)):
                    # The chain calls such a window ``audio_unavailable``: the
                    # offset pushes part of its span outside the audio, so this
                    # offset simply has one window fewer than zero does. Dropping
                    # it keeps every scored window a window the chain has.
                    continue
                collected_decisions[index].append(
                    -1 if abs(first - second) <= 1e-12
                    else int(np.argmax([first, second]))
                )
                collected_truths[index].append(int(features.label))
        if progress is not None:
            progress(trial)
    decisions = [np.asarray(column, dtype=np.int8)
                 for column in collected_decisions]
    truths = [np.asarray(column, dtype=np.int8) for column in collected_truths]
    if not any(len(column) for column in truths):
        raise ValueError("The sweep scored no window.")
    return decisions, truths


def curve(decisions, truths, offsets):
    """Per-offset metrics plus the fold-free parts of the protocol.

    ``decisions[index]`` and ``truths[index]`` are one offset's own window set,
    which can be a window or two smaller than zero's at the extremes of the grid
    (see :func:`sweep`). Every point reports its own ``windows`` count.
    """

    points = []
    for index, offset in enumerate(offsets):
        record = train_kuleuven.metrics(decisions[index], truths[index])
        record["offset_seconds"] = float(offset)
        points.append(record)
    return points


def peak_of(points):
    """Where balanced accuracy is highest, and how flat the top is."""

    values = np.array([point["balanced_accuracy"] for point in points])
    best = int(np.argmax(values))
    ties = np.flatnonzero(np.isclose(values, values[best], atol=1e-12))
    return {
        "offset_seconds": float(points[best]["offset_seconds"]),
        "balanced_accuracy": float(values[best]),
        "tied_offsets_seconds": [float(points[index]["offset_seconds"]) for index in ties],
        "flat_top": len(ties) > 1,
    }


def _crossing(points, level, side, peak_index):
    """First grid point past ``level`` walking outward from the peak; None if none.

    "The crossing on the left" is the first one to the left *of the peak*, so the
    walk starts at the peak and moves outward. Walking inward from the end of the
    grid instead finds the opposite side's crossing on any single-peaked curve,
    and because :func:`width_of` subtracts the two, that version reported a
    *negative* width for both contracts.

    A point that sits exactly on the level counts as a crossing, because a
    decoder that is exactly at chance is not "above chance everywhere".
    """

    offsets = np.array([point["offset_seconds"] for point in points])
    values = np.array([point["balanced_accuracy"] for point in points])
    order = np.argsort(offsets)
    offsets, values = offsets[order], values[order]
    step = -1 if side == "left" else 1
    index = peak_index
    if values[index] == level:
        return float(offsets[index])
    while 0 <= index + step < len(values):
        outer = index + step
        if values[outer] == level:
            return float(offsets[outer])
        if (values[outer] - level) * (values[index] - level) < 0:
            fraction = (level - values[index]) / (values[outer] - values[index])
            return float(offsets[index]
                         + fraction * (offsets[outer] - offsets[index]))
        index = outer
    return None


def width_of(points, peak):
    """Peak width at half depth towards chance, and the extent above chance.

    "Half depth" is halfway from the peak's balanced accuracy down to 0.5, the
    chance line of a balanced accuracy, and not down to the majority rate: the
    majority rate is not a chance line for a balanced metric.
    """

    offsets = np.array([point["offset_seconds"] for point in points])
    peak_index = int(np.argmin(np.abs(offsets - peak["offset_seconds"])))
    level = CHANCE_BALANCED + (peak["balanced_accuracy"] - CHANCE_BALANCED) / 2
    left_half = _crossing(points, level, "left", peak_index)
    right_half = _crossing(points, level, "right", peak_index)
    left_null = _crossing(points, CHANCE_BALANCED, "left", peak_index)
    right_null = _crossing(points, CHANCE_BALANCED, "right", peak_index)
    half = None
    if left_half is not None and right_half is not None:
        half = right_half - left_half
    return {
        "half_depth_level": float(level),
        "left_half_depth_seconds": left_half,
        "right_half_depth_seconds": right_half,
        "half_depth_width_seconds": half,
        "left_chance_seconds": left_null,
        "right_chance_seconds": right_null,
        "chance_width_seconds": (None if left_null is None or right_null is None
                                 else right_null - left_null),
    }


def asymmetry_of(points, peak):
    """Mean balanced accuracy on each side of the peak, and their difference."""

    offsets = np.array([point["offset_seconds"] for point in points])
    values = np.array([point["balanced_accuracy"] for point in points])
    earlier = values[offsets < 0]
    later = values[offsets > 0]
    return {
        "mean_balanced_before_peak_seconds": float(earlier.mean()),
        "mean_balanced_after_peak_seconds": float(later.mean()),
        "mean_difference_after_minus_before": float(later.mean() - earlier.mean()),
        "mean_balanced_all": float(values.mean()),
        "slope_per_100ms_before": _slope(offsets[offsets <= 0], values[offsets <= 0]),
        "slope_per_100ms_after": _slope(offsets[offsets >= 0], values[offsets >= 0]),
    }


def _slope(offsets, values):
    """Least-squares balanced-accuracy change per 100 ms of offset."""

    if len(offsets) < 2:
        return None
    design = np.column_stack([offsets, np.ones(len(offsets))])
    gradient = np.linalg.lstsq(design, values, rcond=None)[0][0]
    return float(gradient * 0.1)


def describe(points):
    """Everything the report needs about one contract's curve."""

    peak = peak_of(points)
    return {"peak": peak, "width": width_of(points, peak),
            "asymmetry": asymmetry_of(points, peak)}


def cost_per_100ms(points, peak):
    """Balanced accuracy lost per 100 ms of misalignment, both directions.

    Read as a difference from the peak over the sweep's own range, which is what
    an operator can act on: "if my alignment is off by 100 ms, I lose this".
    """

    offsets = np.array([point["offset_seconds"] for point in points])
    values = np.array([point["balanced_accuracy"] for point in points])
    cost = {}
    for sign, name in ((-1.0, "negative"), (1.0, "positive")):
        wanted = 0.1 * sign
        available = offsets[np.isclose(offsets, wanted)]
        if len(available):
            index = int(np.argmin(np.abs(offsets - wanted)))
            cost[name] = float(peak["balanced_accuracy"] - values[index])
        else:
            cost[name] = None
    return cost


def corpus_for(key=None):
    """The cached trials of the frozen key, in the order the cache lists them."""

    key = feature_cache.cache_key()[0] if key is None else key
    paths = sorted(feature_cache.cache_directory(key).glob("S*/trial_*.npz"))
    if not paths:
        raise FileNotFoundError(
            f"No cached trials under {feature_cache.cache_directory(key)}."
        )
    return train_kuleuven.Corpus(key, paths)


def converted_path_for(trial):
    """The converted trial one cached trial was built from.

    The cache file is named after the converted trial's stem, so the source path
    is read from the cache rather than reconstructed: the converted files are
    zero-padded (``trial_000.npz``) while ``trial_id`` is not (``"0"``), and
    guessing the padding is exactly the kind of assumption that produces a
    "file not found" in an audit that was supposed to prove alignment.
    """

    relative = trial.path.relative_to(feature_cache.cache_directory(trial.path.parents[1].name))
    return REPO / "datasets" / "AAD-KULeuven" / "converted" / relative


def audit_chain_offset(trial_path, model_path, offsets, history=MAIN_HISTORY,
                       limit_seconds=60.0):
    """Compare the cached pairing against the raw chain, window by window.

    This is the measurement the pairing rests on, not a re-derivation of it. The
    chain path re-reads the audio itself, so its ``aligned.envelopes`` are what
    the offset *means*; the cached path interpolates an envelope that is already
    on the chain's grid. For every window the chain considers valid, this reports

    * ``envelope_max_difference``: ``max |chain - cached|`` for the position
      formula the sweep uses (``start + i - offset * rate``), and
    * ``envelope_mirror_max_difference``: the same number for the mirror image
      (``start + i + offset * rate``), which is what an inverted sign would
      produce. A pairing is pinned only when the first is floating-point noise
      and the second is not.

    The two paths are also scored with the deployed model, so a pairing error
    that survives the envelope check cannot hide in the decision comparison.
    Only windows whose whole span lies inside ``limit_seconds`` are scored, so
    the chain pass can stop early without changing the window set.
    """

    from nova2026.auditory.data import load_trial
    from nova2026.auditory.decoder import RidgeDecoder

    from .runner import labels_for_window, replay_windows

    trial = load_trial(trial_path)
    model = RidgeDecoder.load(model_path)
    if model.training_info.get("history_seconds") != history:
        raise ValueError("The audit model was not fitted at the sweep's window length.")
    rate = feature_cache.sample_rate()
    length = round(history * rate)
    count = length - model.config.lag_samples
    features = feature_cache.load(
        REPO / "datasets" / "auditory_features" / EXPECTED_CACHE_KEY
        / trial_path.parent.name / trial_path.name
    )
    signal = features.signal("64ch").astype(float)
    origin = float(trial.timestamps[0])
    rows = []
    for offset in offsets:
        record = {"offset_seconds": float(offset), "windows": 0,
                  "worst_envelope_window_start": None,
                  "envelope_max_difference": 0.0,
                  "envelope_mirror_max_difference": 0.0,
                  "score_max_difference": 0.0,
                  "score_mirror_max_difference": 0.0,
                  "chain_envelope_mean_abs": 0.0}
        chain_scores = []
        cached_scores = []
        truths = []
        for window in replay_windows(trial, model.config, history, audio_offset=offset):
            if float(window.timestamps[-1]) - origin > limit_seconds:
                break
            if not window.valid:
                continue
            labels = np.unique(labels_for_window(trial, window))
            if len(labels) != 1 or labels[0] < 0:
                continue
            start = int(round((float(window.timestamps[0]) - origin) * rate))
            chain_envelopes = np.asarray(window.envelopes, dtype=float)[:count]
            cached = paired_envelope(
                features.envelopes, positions_for(start, count, offset, rate)
            )
            mirror = paired_envelope(
                features.envelopes, positions_for(start, count, -offset, rate)
            )
            if cached is None or mirror is None:
                continue
            difference = float(np.max(np.abs(cached - chain_envelopes)))
            if difference > record["envelope_max_difference"]:
                record["envelope_max_difference"] = difference
                record["worst_envelope_window_start"] = start
            record["envelope_mirror_max_difference"] = max(
                record["envelope_mirror_max_difference"],
                float(np.max(np.abs(mirror - chain_envelopes))),
            )
            record["chain_envelope_mean_abs"] = max(
                record["chain_envelope_mean_abs"],
                float(np.mean(np.abs(chain_envelopes))),
            )
            pair = trial_scores(model, signal, features.envelopes, start,
                                length, np.asarray([offset]), rate)[0]
            mirrored = trial_scores(model, signal, features.envelopes, start,
                                    length, np.asarray([-offset]), rate)[0]
            truth = model.score(window)
            if not (np.isfinite(pair).all() and np.isfinite(mirrored).all()):
                continue
            record["windows"] += 1
            record["score_max_difference"] = max(
                record["score_max_difference"],
                float(np.max(np.abs(pair - truth))),
            )
            record["score_mirror_max_difference"] = max(
                record["score_mirror_max_difference"],
                float(np.max(np.abs(mirrored - truth))),
            )
            chain_scores.append(truth)
            cached_scores.append(pair)
            truths.append(int(labels[0]))
        truth = np.asarray(truths)
        chain = train_kuleuven.metrics(_decisions(chain_scores), truth)
        cached = train_kuleuven.metrics(_decisions(cached_scores), truth)
        record.update({
            "chain_accuracy": chain["accuracy"],
            "cached_accuracy": cached["accuracy"],
            "chain_balanced_accuracy": chain["balanced_accuracy"],
            "cached_balanced_accuracy": cached["balanced_accuracy"],
            "difference": cached["balanced_accuracy"] - chain["balanced_accuracy"],
        })
        rows.append(record)
    return rows


def _decisions(scores):
    """The decision rule step 5 scores with: a tie abstains, otherwise argmax."""

    return np.asarray([-1 if abs(float(s[0]) - float(s[1])) <= 1e-12
                       else int(np.argmax([float(s[0]), float(s[1])]))
                       for s in scores])


def script_sha256():
    """The digest of this file, so a result can name the code that produced it."""

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(RESULTS))
    parser.add_argument("--step", type=float, default=STEP_SECONDS)
    parser.add_argument("--half-range", type=float, default=HALF_RANGE_SECONDS)
    parser.add_argument("--substeps", type=int, default=SUBSTEPS_PER_STEP)
    parser.add_argument("--history", type=float, default=MAIN_HISTORY)
    parser.add_argument("--contracts", nargs="*", default=list(CONTRACTS))
    parser.add_argument("--tolerance", type=float, default=REFERENCE_TOLERANCE,
                        help="how close the zero point must land on step 5's number")
    parser.add_argument("--audit", action="store_true",
                        help="also compare the cached shift against the raw chain")
    parser.add_argument("--json", help="explicit output path")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    key = feature_cache.cache_key()[0]
    if key != EXPECTED_CACHE_KEY:
        raise SystemExit(
            f"Cache key is {key}, not the frozen {EXPECTED_CACHE_KEY}; the sweep "
            "would be measuring a different feature contract. Stop and report."
        )
    offsets = times_grid(args.half_range, args.step, args.substeps)
    corpus = corpus_for(key)
    document = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": "python -B -m scripts.auditory.shift_sweep "
                   f"--out {args.out}" + (" --audit" if args.audit else ""),
        "kind": "held-out-story sweep; window-level; no human-benefit claim",
        "cache_key": key,
        "cache_trials": len(corpus.trials),
        "history_seconds": args.history,
        "alpha": ALPHA,
        "contracts": list(args.contracts),
        "offset_grid_seconds": [float(value) for value in offsets],
        "offset_grid_note": "every point is a multiple of 12.5 ms; the shift is "
                            "interpolated on the envelope grid, not rounded",
        "sign_convention": "positive offset = audio sample zero is later, so the "
                           "reference envelope is read earlier; the same quantity "
                           "as scripts/auditory/runner.py::replay_windows(audio_offset=)",
        "method": {
            "envelope_shift": "cached envelope sampled at (grid - offset); the EEG "
                              "stream and the models are untouched",
            "models": "fitted once per held-out story fold at zero offset, then "
                      "evaluated at every offset (a deployed model's tolerance)",
            "window_grid": "scripts/auditory/train.py::prepare's non-overlapping "
                           "grid, via feature_cache.Features.windows(hop=history)",
            "metrics": "nova2026.auditory.evaluation-compatible: accuracy, balanced "
                       "accuracy, per-class recall and the majority rate",
            "sha256_of_script": script_sha256(),
            "audited_against_chain": bool(args.audit),
        },
        "reference_step5_balanced": REFERENCE_BALANCED,
        "reference_tolerance": args.tolerance,
        "curves": {},
    }
    for contract in args.contracts:
        start = time.time()
        decisions, truths = sweep(corpus, contract, args.history, offsets)
        points = curve(decisions, truths, offsets)
        zero = float(points[int(np.argmin(np.abs(offsets)))]["balanced_accuracy"])
        reference = REFERENCE_BALANCED[contract]
        if abs(zero - reference) > args.tolerance:
            raise SystemExit(
                f"{contract}: the sweep's zero point is {zero:.4f}, but step 5 "
                f"reported {reference:.4f} (tolerance {args.tolerance:g}). The "
                "harness differs from step 5's, so the curve would be measuring "
                "the harness. Stop and report."
            )
        document["curves"][contract] = {
            "contract": contract, "history_seconds": args.history,
            "windows": points[0]["windows"],
            "balanced_accuracy": [point["balanced_accuracy"] for point in points],
            "accuracy": [point["accuracy"] for point in points],
            "majority_accuracy": [point["majority_accuracy"] for point in points],
            "recall": {"0": [point["recall"]["0"] for point in points],
                       "1": [point["recall"]["1"] for point in points]},
            "decided": [point["decided"] for point in points],
            "points": points,
            "zero_point_balanced_accuracy": zero,
            "zero_point_matches_step5": True,
            **describe(points),
            "cost_per_100ms_balanced": cost_per_100ms(points, peak_of(points)),
        }
        print(f"[{time.time() - started:6.1f}s] {contract}: zero point {zero:.4f} "
              f"(step 5 {reference:.4f})", flush=True)
    if args.audit:
        audit_offsets = np.asarray([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
        audited = []
        for subject in ("S1", "S2"):
            trial = next(trial for trial in corpus.trials if trial.subject == subject)
            audited.append({
                "trial": [trial.subject, trial.trial_id],
                "converted": str(converted_path_for(trial).relative_to(REPO)),
                "rows": audit_chain_offset(
                    converted_path_for(trial),
                    REPO / "models" / train_kuleuven.MODEL_NAMES["64ch"],
                    audit_offsets, args.history,
                ),
            })
        document["chain_audit"] = {
            "model": "models/" + train_kuleuven.MODEL_NAMES["64ch"],
            "note": "the chain path resamples the raw audio itself and needs no "
                    "interpolation; the cached path interpolates the aligned envelope",
            "offset_convention": "cached = env(start + i - offset * rate); "
                                 "mirror = env(start + i + offset * rate)",
            "trials": audited,
        }
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = Path(args.json) if args.json else Path(args.out) / f"aad_shift_sweep_{stamp}.json"
    md_path = Path(args.out) / f"aad_shift_sweep_{stamp}.md"
    png_path = Path(args.out) / f"aad_shift_sweep_{stamp}.png"
    document["elapsed_seconds"] = round(time.time() - started, 1)
    guard_outputs([json_path, md_path, png_path], force=args.force)
    if args.force:
        png_path.unlink(missing_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    write_report(document, md_path)
    if plot(document, png_path):
        document["plot"] = str(png_path.name)
    print(f"[{time.time() - started:6.1f}s] wrote {json_path.name}, "
          f"{md_path.name}, {png_path.name}", flush=True)
    return 0


def plot(document, path):
    """Draw the decay curves; a missing matplotlib is not a failure."""

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - environment without matplotlib
        return False
    offsets = document["offset_grid_seconds"]
    figure, axes = plt.subplots(figsize=(7.0, 4.0))
    for contract, entry in document["curves"].items():
        axes.plot(offsets, entry["balanced_accuracy"], marker="o", markersize=3,
                  label=f"{contract} balanced")
        axes.plot(offsets, entry["accuracy"], linestyle="--", linewidth=1,
                  label=f"{contract} accuracy")
    axes.axhline(0.5, color="grey", linewidth=1, label="balanced chance 0.5")
    axes.axhline(document["curves"][next(iter(document["curves"]))]["majority_accuracy"][0],
                 color="black", linewidth=1, linestyle=":", label="majority rate")
    axes.axvline(0.0, color="grey", linewidth=0.8)
    axes.set_xlabel("audio offset (s); positive = audio later")
    axes.set_ylabel("window-level accuracy")
    axes.set_title("Decoder accuracy against audio/EEG timing offset")
    axes.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return True


def write_report(document, path):
    """A short human-readable version of the same run's JSON."""

    lines = [
        "# Audio/EEG offset sweep: what a timing error costs the decoder",
        "",
        f"Generated `{document['generated_at']}` by `{document['command']}`.",
        "",
        f"Cache `{document['cache_key']}`, {document['cache_trials']} trials, "
        f"window {document['history_seconds']:g} s, alpha {document['alpha']:g}, "
        f"held-out-story folds. Sign convention: {document['sign_convention']}.",
        "The models are fitted at zero offset and then evaluated at every offset, "
        "so the curve is a deployed model's tolerance and not a re-training result.",
        "",
        "## The curves",
        "",
        "| offset (ms) | 64ch balanced | 64ch acc. | 20ch balanced | 20ch acc. | majority |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    offsets = document["offset_grid_seconds"]
    first = next(iter(document["curves"].values()))
    for index, offset in enumerate(offsets):
        cells = [f"{offset * 1000:+.1f}"]
        for contract in document["curves"]:
            entry = document["curves"][contract]
            cells.append(f"{entry['balanced_accuracy'][index]:.4f}")
            cells.append(f"{entry['accuracy'][index]:.4f}")
        cells.append(f"{first['majority_accuracy'][index]:.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Peak, width and asymmetry", "",
              "| contract | peak (ms) | peak balanced | half-depth width (ms) | "
              "chance width (ms) | mean(before) | mean(after) | after-before |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for contract, entry in document["curves"].items():
        peak, width, shape = entry["peak"], entry["width"], entry["asymmetry"]
        half = width["half_depth_width_seconds"]
        chance = width["chance_width_seconds"]
        half_text = "n/a" if half is None else f"{half * 1000:.0f}"
        chance_text = "n/a" if chance is None else f"{chance * 1000:.0f}"
        lines.append(
            f"| {contract} | {peak['offset_seconds'] * 1000:+.1f} | "
            f"{peak['balanced_accuracy']:.4f} | {half_text} | {chance_text} | "
            f"{shape['mean_balanced_before_peak_seconds']:.4f} | "
            f"{shape['mean_balanced_after_peak_seconds']:.4f} | "
            f"{shape['mean_difference_after_minus_before']:+.4f} |"
        )
    lines += ["", "## Cost per 100 ms of misalignment", "",
              "| contract | early by 100 ms | late by 100 ms | slope before (per 100 ms) | "
              "slope after (per 100 ms) |", "| --- | --- | --- | --- | --- |"]
    for contract, entry in document["curves"].items():
        cost = entry["cost_per_100ms_balanced"]
        shape = entry["asymmetry"]
        early = cost.get("negative")
        late = cost.get("positive")
        before = shape["slope_per_100ms_before"]
        after = shape["slope_per_100ms_after"]
        lines.append(
            f"| {contract} | {'n/a' if early is None else f'{early:+.4f}'} | "
            f"{'n/a' if late is None else f'{late:+.4f}'} | "
            f"{'n/a' if before is None else f'{before:+.4f}'} | "
            f"{'n/a' if after is None else f'{after:+.4f}'} |"
        )
    if "chain_audit" in document:
        audit = document["chain_audit"]
        lines += ["", "## Cached pairing vs the raw chain, window by window", "",
                  f"Model `{audit['model']}`. {audit['note']}. "
                  f"{audit['offset_convention']}.", "",
                  "`max |Δ|` is over the first `length - lag_samples` ROWS of every window "
                  "the chain calls valid, in envelope units (and then in correlation "
                  "units); the mirror column is the same measurement for the opposite "
                  "sign. The pairing is pinned when the first column is at the storage "
                  "precision of the cache and the second is orders of magnitude larger.", ""]
        for entry in audit["trials"]:
            lines += [f"Trial {entry['trial'][0]}/{entry['trial'][1]} "
                      f"(`{entry['converted']}`)", "",
                      "| offset (ms) | windows | max \\|Δenvelope\\| | mirror | max \\|Δscore\\| | "
                      "mirror | chain balanced | cached balanced | difference |",
                      "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
            for row in entry["rows"]:
                lines.append(
                    f"| {row['offset_seconds'] * 1000:+.1f} | {row['windows']} | "
                    f"{row['envelope_max_difference']:.2e} | "
                    f"{row['envelope_mirror_max_difference']:.2e} | "
                    f"{row['score_max_difference']:.2e} | "
                    f"{row['score_mirror_max_difference']:.2e} | "
                    f"{row['chain_balanced_accuracy']:.4f} | "
                    f"{row['cached_balanced_accuracy']:.4f} | {row['difference']:+.4f} |"
                )
            lines.append("")
    lines += [
        "",
        "## What these numbers do and do not say",
        "",
        f"- The sweep reproduces step 5's zero point within "
        f"{document['reference_tolerance']:g} balanced accuracy "
        f"({document['reference_step5_balanced']}), so the curve is step 5's "
        "harness with one parameter moved.",
        "- The absolute level is weak (about 0.62 balanced against a 0.50 chance "
        "line for a balanced metric, and below the majority rate on raw accuracy). "
        "A weak decoder's decay curve is not a strong decoder's decay curve.",
        "- Windows overlap in nothing but the trial: the grid is step 5's "
        "non-overlapping one, so the point count is windows, not independent "
        "subjects, and the per-point confidence is wide.",
        "- One dataset, one window length, one alpha. The offsets are applied to "
        "the reference envelope; a real misalignment could also drift with time, "
        "which this experiment does not model.",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
