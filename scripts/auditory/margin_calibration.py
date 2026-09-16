"""Calibrate the controller's decision margin against coverage, not against hope.

Step 8 ran the whole chain end to end on one real trial and published 496 frames:
465 ``uncertain``, 19 ``unavailable``, 12 ``A``. Ninety-four percent of the demo
said nothing, because :class:`~nova2026.auditory.controller.AttentionController`
refuses to commit while ``abs(score[0] - score[1]) < MIN_MARGIN`` and
``MIN_MARGIN = 0.5`` had never been measured against anything. Step 5.5 measured
what a *timing offset* costs; it never touched the margin. This script measures
the margin.

What it measures, on held-out stories only:

* **coverage** -- the fraction of scored windows, and of published frames, that
  reach a decision instead of ``uncertain``;
* **accuracy and balanced accuracy on the decided frames**, plus per-class
  recall, because the labels are 66.1% A by time and a margin that commits
  mostly on A windows raises raw accuracy while leaving balanced accuracy at
  chance;
* the same numbers for **every margin in a ladder** crossed with **window
  length**, so an operating point can be chosen instead of guessed;
* **switching behaviour** at each point: how often the decision changes to a
  wrong candidate while the truth did not change, and how long a true switch
  takes to be reported, using
  :func:`nova2026.auditory.evaluation.selection_metrics`.

What it refuses to do:

* **touch the default.** ``MIN_MARGIN`` stays ``0.5``; the ladder reports it as
  one point among many. A calibrated margin is a run-policy choice, and this
  script's job is to publish the trade-off, not to move the knob.
* **use the deployed model.** ``models/auditory_kuleuven.npz`` was fitted on all
  320 trials, so nothing it scores is held out. Every number here comes from a
  model refitted inside a fold whose held-out story is excluded, and the story
  guard from :mod:`nova2026.auditory.evaluation` is run against each fold.
* **re-fit per margin.** The decoder is fitted once per fold per window length;
  the margin is applied afterwards as a decision rule. Moving the margin
  therefore cannot move the decoder, which is what makes the sweep a sweep of
  one knob.
* **report accuracy without its null.** Every row carries the majority rate of
  its own decided subset, and the fraction of frames that are ``unavailable``
  (warm-up and the window-length wait) next to the fraction that are
  ``uncertain``.

The scoring path is a documented equivalent of ``model.score(window)``: the
reconstruction of a whole trial is one lag-weighted pass over the signal, and a
window's score is the correlation of that series with the envelope over the same
rows. :func:`pin_scores` checks the equivalence against the decoder itself, on
real windows, in every fold, and the worst disagreement is reported.

    python -B -m scripts.auditory.margin_calibration --out results

Writes ``results/aad_margin_calibration_<YYYYMMDD-HHMMSS>.json`` and ``.md``.
"""

import argparse
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from nova2026.auditory.config import MIN_MARGIN
from nova2026.auditory.controller import AttentionController
from nova2026.auditory.data import AttentionEstimate
from nova2026.auditory.evaluation import assert_held_out, selection_metrics

from . import aad_ridge, feature_cache, train_kuleuven
from .kuleuven_contract import RESULTS

# The ladder has to contain the value it is judging, or it is not a calibration
# of that value. 0.60 is past the current default on purpose: if the honest
# answer were "even 0.5 is too permissive", the curve has to show it.
MARGIN_LADDER = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60)
HISTORIES = train_kuleuven.CURVE_HISTORIES
ALPHA = train_kuleuven.DEFAULT_ALPHA
# The session's own frame cadence and warm-up, from ``RunPolicy``. Restating
# them here would let the two drift, so ``main`` asserts they still agree with
# the session's defaults and the JSON records what was used.
FRAME_SECONDS = 0.25
WARMUP_SECONDS = 2.0
MIN_SWITCH_WINDOWS = 3
MAX_AGE = 3.0
ATTENUATION_DB = 6.0
PIN_WINDOWS = 4
# A correlation difference computed two ways disagrees only by rounding; the
# tolerance is loose enough for reassociated sums and tight enough that a real
# pairing mistake (a sign, an offset sample, a dropped lag) cannot pass.
PIN_TOLERANCE = 1e-9

A, B, UNCERTAIN, UNAVAILABLE = 0, 1, -1, -2


def reconstruction_series(model, signal):
    """``w . design`` for every row of one trial, without building the design.

    ``RidgeDecoder.score`` builds a ``(rows, channels * (lag + 1))`` matrix per
    window and multiplies it by the weights. The design is a lag stack of one
    signal, so the same number comes out of summing ``lag + 1`` matrix-vector
    products over the whole recording once -- and every window then reads a
    slice of it. The arithmetic is the decoder's, reassociated.
    """

    lag = model.config.lag_samples
    normalized = (np.asarray(signal, dtype=float) - model.mean) / model.scale
    rows = len(normalized) - lag
    if rows < 2:
        raise ValueError("A trial must be longer than the decoder's lags.")
    weights = model.weights.reshape(lag + 1, len(model.mean))
    series = np.zeros(rows)
    for offset in range(lag + 1):
        series += normalized[offset : offset + rows] @ weights[offset]
    return series


def correlation_scores(series, envelopes, start, length, lag):
    """One window's two correlations: ``model.score``'s arithmetic, restated."""

    stop = start + length - lag
    reconstruction = series[start:stop]
    reconstruction = reconstruction - reconstruction.mean()
    scores = np.zeros(2)
    for candidate in range(2):
        envelope = np.asarray(envelopes[start:stop, candidate], dtype=float)
        envelope = envelope - envelope.mean()
        denominator = np.linalg.norm(reconstruction) * np.linalg.norm(envelope)
        if denominator > 1e-12:
            scores[candidate] = float(np.dot(reconstruction, envelope) / denominator)
    return scores


def frame_grid(times, frame_seconds=FRAME_SECONDS):
    """The published frame instants of one trial, on the session's cadence."""

    if len(times) < 2:
        raise ValueError("A trial needs at least two samples to publish a frame.")
    count = int(np.floor((times[-1] - times[0]) / frame_seconds + 1e-9)) + 1
    return times[0] + np.arange(count) * frame_seconds


def scored_windows(model, features, contract, history, pin=PIN_WINDOWS):
    """Score every window the chain would emit, and check the fast path.

    Returns the accepted window starts, their evidence ends (the last sample of
    the window, which is when the chain can emit it), and their scores. The
    scoring grid is the chain's own one-second hop, so the controller sees the
    stream a live session would see; it is *not* the non-overlapping grid the
    fold was fitted on.
    """

    signal = features.signal(contract).astype(float)
    series = reconstruction_series(model, signal)
    lag = model.config.lag_samples
    length = round(history * feature_cache.sample_rate())
    starts, scores = [], []
    for start, _ in features.windows(contract, history):
        starts.append(int(start))
        scores.append(correlation_scores(series, features.envelopes, start, length, lag))
    starts = np.asarray(starts, dtype=np.int64)
    scores = np.asarray(scores, dtype=float).reshape(-1, 2)
    worst = 0.0
    if pin and len(starts):
        step = max(1, len(starts) // pin)
        for position in range(0, len(starts), step)[:pin]:
            window = train_kuleuven.window_at(features, contract, history, starts[position])
            reference = np.asarray(model.score(window), dtype=float)
            worst = max(worst, float(np.max(np.abs(reference - scores[position]))))
    arrivals = np.asarray(features.times[starts + length - 1], dtype=float)
    return starts, arrivals, scores, worst


def replay_windows(scores, arrivals, margin, *, controller_kwargs=None):
    """Drive a real ``AttentionController`` over one trial's scored windows.

    This is the plumb-through, not a restatement of it: every state comes out of
    :meth:`AttentionController.update`, so a change in the controller's rules
    moves this sweep with it. ``now`` is the window's own evidence end, i.e. the
    optimistic zero-latency case; the offload queue adds at most one window of
    latency and ``max_age`` is three times the hop, so latency cannot change the
    coverage reported here by more than a warm-up window.
    """

    kwargs = dict(controller_kwargs or {})
    controller = AttentionController(margin=margin, **kwargs)
    states = np.full(len(scores), UNCERTAIN, dtype=np.int8)
    for index in range(len(scores)):
        evidence_end = float(arrivals[index])
        score = np.asarray(scores[index], dtype=float)
        controller.update(
            AttentionEstimate(score, evidence_end, evidence_end, True, ()), evidence_end
        )
        if controller.selected is not None:
            states[index] = int(controller.selected)
    return states


def frame_decisions(states, arrivals, frame_times, *, max_age=MAX_AGE,
                    warmup_seconds=WARMUP_SECONDS):
    """Map per-window states onto the session's published frames.

    Mirrors ``AttentionSession._frame``: before any evidence the frame is
    ``unavailable`` (``no_evidence``), during warm-up it is ``unavailable``,
    evidence older than ``max_age`` is ``unavailable`` (``evidence_stale``), and
    otherwise the frame reports the controller's current selection or
    ``uncertain``.
    """

    decisions = np.full(len(frame_times), UNAVAILABLE, dtype=np.int8)
    if not len(arrivals):
        return decisions
    index = np.searchsorted(arrivals, frame_times, side="right") - 1
    fresh = index >= 0
    age = np.where(fresh, frame_times - arrivals[np.clip(index, 0, None)], np.inf)
    warm = (frame_times - frame_times[0]) < warmup_seconds
    live = fresh & ~warm & (age <= max_age)
    usable = live & (index >= 0)
    if np.any(usable):
        chosen = states[index[usable]]
        decisions[usable] = np.where(chosen >= 0, chosen, UNCERTAIN)
    return decisions


def decision_metrics(decisions, truths):
    """Coverage, accuracy, balanced accuracy and both per-class reads.

    Three numbers are easy to confuse and are therefore all reported:

    ``accuracy_decided``
        correct / decided. The number a demo's audience reads.
    ``balanced_accuracy_decided``
        the mean of the two per-class hit rates *within* the decided frames,
        with a class that has **no decided frames counted as 0.0** rather than
        dropped. Dropping it would score a margin that only ever commits on
        class A -- and is always right there -- as a perfect 1.0; counting it as
        zero scores that margin 0.5, which is what it deserves. A class with no
        decided frames is still reported as ``None`` in ``recall_decided``, so
        the reason for the 0.5 is visible.
    ``recall_over_all``
        per-class hits divided by all frames of that class, so an abstention
        counts as a miss. This is the number step 5 publishes, kept for
        comparability, and its mean is the conventional balanced accuracy.
    """

    decisions = np.asarray(decisions)
    truths = np.asarray(truths)
    known = np.isin(truths, [A, B])
    decided = known & np.isin(decisions, [A, B])
    correct = decided & (decisions == truths)
    total = int(known.sum())
    result = {
        "frames": total,
        "decided": int(decided.sum()),
        "coverage": float(decided.sum() / total) if total else None,
        "accuracy_decided": (float(correct.sum() / decided.sum())
                             if decided.any() else None),
        "unavailable": int(np.count_nonzero(known & (decisions == UNAVAILABLE))),
        "uncertain": int(np.count_nonzero(known & (decisions == UNCERTAIN))),
        "class_frames": {str(value): int(np.count_nonzero(known & (truths == value)))
                         for value in (A, B)},
        "decided_class_frames": {str(value): int(np.count_nonzero(decided & (truths == value)))
                                 for value in (A, B)},
    }
    conditional, unconditional, balanced_parts = {}, {}, []
    for value in (A, B):
        mask = known & (truths == value)
        within = decided & (truths == value)
        hit = (float(np.count_nonzero(correct & (truths == value)) / within.sum())
               if within.any() else None)
        conditional[str(value)] = hit
        balanced_parts.append(0.0 if hit is None else hit)
        unconditional[str(value)] = (float(np.count_nonzero(correct & (truths == value))
                                           / mask.sum()) if mask.any() else None)
    result["recall_decided"] = conditional
    result["recall_over_all"] = unconditional
    result["balanced_accuracy_decided"] = float(np.mean(balanced_parts))
    if decided.any():
        counts = [result["decided_class_frames"][str(value)] for value in (A, B)]
        result["majority_decided"] = float(max(counts) / decided.sum())
        result["accuracy_all_frames"] = float(correct.sum() / total) if total else None
    else:
        result["majority_decided"] = None
        result["accuracy_all_frames"] = (0.0 if total else None)
    counts = [result["class_frames"][str(value)] for value in (A, B)]
    result["majority_over_all"] = float(max(counts) / total) if total else None
    return result


def window_metrics(states, truths):
    """The window-level read: one decision opportunity per emitted window."""

    states = np.asarray(states)
    truths = np.asarray(truths)
    total = len(states)
    decided = np.isin(states, [A, B])
    correct = decided & (states == truths)
    record = {
        "windows": int(total),
        "decided": int(decided.sum()),
        "coverage": float(decided.sum() / total) if total else None,
        "accuracy_decided": (float(correct.sum() / decided.sum())
                             if decided.any() else None),
    }
    present = []
    for value in (A, B):
        within = decided & (truths == value)
        hit = float(np.count_nonzero(correct & (truths == value)) / within.sum()) \
            if within.any() else None
        record[f"recall_decided_{'A' if value == A else 'B'}"] = hit
        if hit is not None:
            present.append(hit)
    record["balanced_accuracy_decided"] = float(np.mean(present)) if present else None
    return record


def pooled_selection(records):
    """Pool per-fold ``selection_metrics`` output into one switching summary."""

    known = sum(record["known_seconds"] for record in records)
    covered = sum((record["coverage"] or 0.0) * record["known_seconds"] for record in records)
    delays = [delay for record in records
              for delay in record["reported_switch_delays_seconds"]]
    changes = sum(record["false_selection_changes"] for record in records)
    minutes = known / 60.0
    return {
        "known_seconds": known,
        "coverage": (covered / known) if known else None,
        "false_selection_changes": int(changes),
        "false_selection_changes_per_minute": (changes / minutes) if minutes else None,
        "reported_switch_delays_seconds": delays,
        "reported_switch_delay_median": (float(np.median(delays)) if delays else None),
        "reported_switch_delay_p90": (float(np.percentile(delays, 90)) if delays else None),
        "reported_switch_delay_count": len(delays),
        "missed_switches": sum(record["missed_switches"] for record in records),
        "neutral_seconds": sum(record["neutral_seconds"] for record in records),
    }


def folds_with_models(corpus, contract, history, alpha, progress=None):
    """One held-out-story fold per story, each with a model refitted without it.

    The same fold definition step 5 used (``train_kuleuven.held_out_story_folds``
    over the canonical ``group_key``, which folds ``rep_*`` into its base), the
    same moments arithmetic, and the same leakage guard.
    """

    sums = train_kuleuven.moments_pass(corpus, contract, history, progress=progress)
    prepared = []
    for fold in train_kuleuven.held_out_story_folds(corpus):
        training = train_kuleuven.training_moments(sums, fold["validation"])
        provenance = {
            "training": [(t.subject, t.trial_id, t.group) for t in fold["training"]],
            "held_out": [(t.subject, t.trial_id, t.group) for t in fold["validation"]],
            "scheme": "held_out_story", "fold": fold["name"], "contract": contract,
            "history_seconds": history, "alpha": alpha, "cache_key": corpus.key,
            "labels_in_inference_path": False,
        }
        model = aad_ridge.fit(training, corpus.contract_of(contract, history), alpha,
                              training_info=provenance)
        for trial in fold["validation"]:
            assert_held_out(trial, model)
        prepared.append({"name": fold["name"], "model": model,
                         "held_out": fold["validation"]})
    return prepared, sums


def stream_of(corpus, contract, history, progress=None):
    """Score every held-out trial once and keep the streams the sweep needs."""

    key = corpus.key
    prepared, sums = folds_with_models(corpus, contract, history, ALPHA, progress)
    trials, worst = [], 0.0
    for fold in prepared:
        for trial in fold["held_out"]:
            features = feature_cache.load(trial.path, key)
            starts, arrivals, scores, pin = scored_windows(fold["model"], features,
                                                           contract, history)
            if not len(starts):
                raise ValueError(f"{trial.path} produced no window at {history:g}s.")
            worst = max(worst, pin)
            times = frame_grid(np.asarray(features.times, dtype=float))
            index = np.searchsorted(features.times, times, side="right") - 1
            truths = np.asarray(features.labels, dtype=int)[index]
            trials.append({"fold": fold["name"], "subject": trial.subject,
                           "trial_id": trial.trial_id, "group": trial.group,
                           "label": trial.label, "starts": starts,
                           "arrivals": arrivals, "scores": scores,
                           "frame_times": times, "frame_truths": truths})
            if progress is not None:
                progress(len(trials), trial)
    if worst > PIN_TOLERANCE:
        raise ValueError(
            f"The fast scoring path disagrees with the decoder by {worst:g}, "
            f"above the {PIN_TOLERANCE:g} tolerance; the sweep would be "
            "measuring its own arithmetic."
        )
    return {"contract": contract, "history_seconds": history, "trials": trials,
            "folds": [fold["name"] for fold in prepared],
            "fold_sizes": {fold["name"]: len(fold["held_out"]) for fold in prepared},
            "pin_max_abs_score_difference": worst,
            "training_trials_per_fold": {fold["name"]: len(corpus.trials) - len(fold["held_out"])
                                         for fold in prepared}}


def trial_outcome(trial, margin):
    """Per-trial window states, frame decisions and switching record."""

    states = replay_windows(trial["scores"], trial["arrivals"], margin,
                            controller_kwargs={"max_age": MAX_AGE,
                                               "min_switch_windows": MIN_SWITCH_WINDOWS,
                                               "attenuation_db": ATTENUATION_DB})
    decisions = frame_decisions(states, trial["arrivals"], trial["frame_times"])
    choices = np.where(decisions == UNAVAILABLE, UNCERTAIN, decisions)
    switching = selection_metrics(trial["frame_times"], choices, trial["frame_truths"],
                                  end_time=float(trial["frame_times"][-1] + FRAME_SECONDS))
    decided = np.isin(decisions, [A, B])
    first = float(trial["frame_times"][np.argmax(decided)] - trial["frame_times"][0]) \
        if decided.any() else None
    flips = int(np.count_nonzero((decisions[1:] != decisions[:-1])
                                 & np.isin(decisions[1:], [A, B])
                                 & np.isin(decisions[:-1], [A, B])))
    return {"states": states, "decisions": decisions, "switching": switching,
            "first_decision_seconds": first, "flips": flips,
            "window_truths": np.full(len(states), trial["label"], dtype=int)}


def sweep_margins(stream, margins):
    """Every margin in the ladder, on the same models and the same windows."""

    table = {}
    for margin in margins:
        window_states, window_truths, decisions, truths = [], [], [], []
        switching_records = []
        firsts, flips = [], 0
        by_fold = {}
        for trial in stream["trials"]:
            outcome = trial_outcome(trial, margin)
            window_states.append(outcome["states"])
            window_truths.append(outcome["window_truths"])
            decisions.append(outcome["decisions"])
            truths.append(trial["frame_truths"])
            switching_records.append(outcome["switching"])
            flips += outcome["flips"]
            if outcome["first_decision_seconds"] is not None:
                firsts.append(outcome["first_decision_seconds"])
            fold = by_fold.setdefault(trial["fold"], ([], []))
            fold[0].append(outcome["decisions"])
            fold[1].append(trial["frame_truths"])
        frames = decision_metrics(np.concatenate(decisions), np.concatenate(truths))
        windows = window_metrics(np.concatenate(window_states),
                                 np.concatenate(window_truths))
        switching = pooled_selection(switching_records)
        switching["frames_decided_to_undecided"] = int(np.count_nonzero(
            np.isin(np.concatenate(decisions)[:-1], [A, B])
            & ~np.isin(np.concatenate(decisions)[1:], [A, B])))
        per_fold = {
            name: decision_metrics(np.concatenate(pair[0]), np.concatenate(pair[1]))
            for name, pair in sorted(by_fold.items())
        }
        # The folds are the independent unit, so "it works" has to mean "it works
        # on every held-out story", not "it works pooled". A margin that never
        # commits inside one story has no evidence there and does not qualify.
        scored = [record["balanced_accuracy_decided"] for record in per_fold.values()
                  if record["balanced_accuracy_decided"] is not None]
        table[f"{margin:g}"] = {
            "margin": margin,
            "frames": frames,
            "windows": windows,
            "switching": switching,
            "per_fold": per_fold,
            "fold_balanced_min": (float(min(scored)) if scored else None),
            "fold_balanced_max": (float(max(scored)) if scored else None),
            "folds_above_chance": int(sum(value > 0.5 for value in scored)),
            "folds_with_decisions": len(scored),
            "folds": len(per_fold),
            "candidate_flips": flips,
            "first_decision_seconds_mean": (float(np.mean(firsts)) if firsts else None),
            "first_decision_seconds_median": (float(np.median(firsts)) if firsts else None),
            "trials_with_a_decision": len(firsts),
            "trials": len(stream["trials"]),
        }
    return table


def write_report(document, path):
    """The curves, the recommendation, and what they do not say."""

    lines = [
        "# Auditory decision-margin calibration",
        "",
        f"Generated `{document['generated_at']}` by `{document['command']}`.",
        "",
        f"Cache `{document['cache_key']}`, {document['cache_trials']} trials, "
        f"held out by story (`held_out_story_folds`, `rep_*` folded into its "
        f"base): {len(document['folds'])} folds, "
        f"{document['fold_trial_counts'][0]}-{document['fold_trial_counts'][-1]} "
        "held-out trials each.",
        "",
        "The controller is the session's: `AttentionController` with "
        f"`min_switch_windows={MIN_SWITCH_WINDOWS}`, `max_age={MAX_AGE:g}`, "
        f"`attenuation_db={ATTENUATION_DB:g}`, frames every "
        f"{FRAME_SECONDS:g}s, warm-up {WARMUP_SECONDS:g}s. Every state in the "
        "tables below comes out of `AttentionController.update`; the margin is "
        "the only thing that moves between columns.",
        "",
        "Coverage is the fraction of published frames that report A or B. "
        "`balanced (decided)` is the mean of the two per-class hit rates *within* "
        "the decided frames, so a margin that only commits on A reads 0.50 there "
        "however good its raw accuracy looks; `majority (decided)` is the null it "
        "must beat. `unavail` is warm-up plus the window-length wait, which is "
        "not the same failure as `uncertain` and is not counted as coverage.",
        "",
    ]
    for stream in document["streams"]:
        contract, history = stream["contract"], stream["history_seconds"]
        lines += [
            f"## {contract}, {history:g}s windows",
            "",
            f"Pinned against `model.score` on real windows: worst "
            f"|Δscore| = {stream['pin_max_abs_score_difference']:.2e} "
            f"({PIN_TOLERANCE:g} tolerance).",
            "",
            "| margin | coverage | accuracy (decided) | balanced (decided) | "
            "balanced per story (min–max) | stories above chance | "
            "majority (decided) | recall A/B (decided) | recall A/B (all frames) | "
            "windows decided | unavail | uncertain |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for key, row in stream["margins"].items():
            frames, windows = row["frames"], row["windows"]
            lines.append(
                f"| {row['margin']:.2f} | {frames['coverage']:.4f} | "
                f"{_pct(frames['accuracy_decided'])} | "
                f"{_pct(frames['balanced_accuracy_decided'])} | "
                f"{_pct(row['fold_balanced_min'])}–{_pct(row['fold_balanced_max'])} | "
                f"{row['folds_above_chance']}/{row['folds']} | "
                f"{_pct(frames['majority_decided'])} | "
                f"{_pct(frames['recall_decided']['0'])}/"
                f"{_pct(frames['recall_decided']['1'])} | "
                f"{_pct(frames['recall_over_all']['0'])}/"
                f"{_pct(frames['recall_over_all']['1'])} | "
                f"{windows['decided']}/{windows['windows']} "
                f"({windows['coverage']:.3f}) | "
                f"{frames['unavailable']} | {frames['uncertain']} |"
            )
        lines += [
            "",
            "| margin | false selection changes /min | switch delay median (s) | "
            "switch delays reported | missed switches | candidate flips | "
            "frames decided→undecided | first decision (s) |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for key, row in stream["margins"].items():
            switching = row["switching"]
            lines.append(
                f"| {row['margin']:.2f} | "
                f"{_num(switching['false_selection_changes_per_minute'], 3)} | "
                f"{_num(switching['reported_switch_delay_median'], 2)} | "
                f"{switching['reported_switch_delay_count']} | "
                f"{switching['missed_switches']} | {row['candidate_flips']} | "
                f"{switching['frames_decided_to_undecided']} | "
                f"{_num(row['first_decision_seconds_median'], 1)} |"
            )
        lines.append("")
    lines += [
        "## Recommendation",
        "",
        f"Rule, fixed before the curve was read: {document['recommendation']['rule']}.",
        "",
        document["recommendation"]["text"],
        "",
        "The rule's answer at each window length, so the choice can be moved "
        "without re-running the sweep:",
        "",
        "| window (s) | margin | coverage | accuracy (decided) | balanced (decided) | "
        "per story | first decision (s) | false changes /min |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for history, point in document["recommendation"]["per_window"].items():
        if point is None:
            lines.append(f"| {history} | none qualified | | | | | | |")
            continue
        lines.append(
            f"| {history} | {point['margin']:.2f} | {point['coverage']:.4f} | "
            f"{_pct(point['accuracy_decided'])} | "
            f"{_pct(point['balanced_accuracy_decided'])} | "
            f"{_pct(point['fold_balanced_min'])}–{_pct(point['fold_balanced_max'])} "
            f"({point['folds_above_chance']}) | "
            f"{_num(point['first_decision_seconds_median'], 1)} | "
            f"{_num(point['false_selection_changes_per_minute'], 3)} |"
        )
    lines += ["",
        "## What these numbers do not say",
        "",
        "- They are window- and frame-level, from one dataset (KU Leuven replay) "
        "with the authors' own stimuli. They say nothing about the ANT test set "
        "and nothing about a live participant.",
        "- The unit of every count is a *window* or a *frame*, never a subject. "
        "The story folds are the independent unit (four of them), so the "
        "differences between neighbouring margins are well inside the variation "
        "between stories and must not be read as a ranking.",
        "- A margin tuned here is a demo operating point, not evidence of "
        "decoding quality. The decoder's own numbers are step 5's.",
        "- Frames overlap their neighbours (one-second hop on a "
        f"{document['streams'][0]['history_seconds']:g}s window), so consecutive "
        "frames are far from independent and the effective sample size is much "
        "smaller than the frame count.",
        "- Latency is modelled as zero: the controller is updated with `now` "
        "equal to the window's evidence end. A real offload queue adds at most "
        "one window.",
        "- `unavailable` frames (warm-up and the wait for the first full window) "
        "are reported separately and are not `uncertain`; a longer window buys "
        "accuracy at the price of a longer silent start.",
        "",
        "## Plumbing (for the step that runs the demo)",
        "",
        document["recommendation"]["plumbing"],
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _pct(value):
    return "n/a" if value is None else f"{value:.4f}"


def _num(value, digits):
    return "n/a" if value is None else f"{value:.{digits}f}"


COVERAGE_TOLERANCE = 0.02
RULE = ("a point qualifies if it commits inside every held-out story with "
        "balanced accuracy above chance in each, and if its accuracy on decided "
        "frames beats the majority rate of its own decided subset; among the "
        "qualifying points of a window length, keep the best coverage and then "
        "the highest balanced accuracy within "
        f"{COVERAGE_TOLERANCE:g} of it. Coverage is the price; the balanced "
        "accuracy of the committed frames is what is bought")
DEPLOYED_MODEL = "models/auditory_kuleuven.npz"


def qualifies(row):
    """Whether one (window length, margin) point earned its place."""

    return (
        row["folds_with_decisions"] == row["folds"]
        and row["folds_above_chance"] == row["folds"]
        and row["frames"]["accuracy_decided"] is not None
        and row["frames"]["majority_decided"] is not None
        and row["frames"]["accuracy_decided"] >= row["frames"]["majority_decided"]
    )


def front_runner(rows):
    """The point a window length is judged by: coverage first, then quality.

    Both halves matter. Maximizing accuracy alone picks the extreme margin,
    which decides twice in a whole corpus; maximizing coverage alone picks
    "commit on everything", which is not a decision rule. The tolerance is what
    turns the curve into an operating point: inside it, take the better
    committed accuracy and the stricter margin.
    """

    candidates = [row for row in rows if qualifies(row)]
    if not candidates:
        return None
    best = max(row["frames"]["coverage"] for row in candidates)
    front = [row for row in candidates
             if row["frames"]["coverage"] >= best - COVERAGE_TOLERANCE]
    return max(front, key=lambda row: (row["frames"]["balanced_accuracy_decided"],
                                       row["margin"]))


def point_of(row, history):
    """A compact record of one operating point, ready for the report."""

    if row is None:
        return None
    frames, switching = row["frames"], row["switching"]
    return {
        "history_seconds": history, "margin": row["margin"],
        "coverage": frames["coverage"], "accuracy_decided": frames["accuracy_decided"],
        "balanced_accuracy_decided": frames["balanced_accuracy_decided"],
        "majority_decided": frames["majority_decided"],
        "fold_balanced_min": row["fold_balanced_min"],
        "fold_balanced_max": row["fold_balanced_max"],
        "folds_above_chance": f"{row['folds_above_chance']}/{row['folds']}",
        "recall_over_all": frames["recall_over_all"],
        "unavailable_frames": frames["unavailable"],
        "uncertain_frames": frames["uncertain"],
        "false_selection_changes_per_minute":
            switching["false_selection_changes_per_minute"],
        "candidate_flips": row["candidate_flips"],
        "first_decision_seconds_median": row["first_decision_seconds_median"],
    }


def deployed_window():
    """Which window length the saved model can actually run, read from it.

    The window length is part of the decoder's contract, so a session cannot
    choose it: it can only use a model fitted at that length. Reading the model
    that the demo replays with is therefore the difference between a
    recommendation that can be run tomorrow and one that cannot.
    """

    from .kuleuven_contract import REPO

    path = REPO / DEPLOYED_MODEL
    if not path.is_file():
        return None, None
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
    contract = metadata["contract"]
    channels = len(contract["eeg_channels"])
    return float(contract["window_seconds"]), ("64ch" if channels == 64 else "20ch")


def recommend(document, contract):
    """Name the operating point the plan asked for, and what it costs.

    The rule was fixed before the curve was read (see :data:`RULE`): a rule
    chosen after seeing the numbers is not a rule. It is applied to every window
    length in the run, and separately to the window length the saved model can
    actually run, because those can differ -- and a recommendation that needs a
    model nobody has is not a recommendation.
    """

    streams = [stream for stream in document["streams"]
               if stream["contract"] == contract]
    per_window = {}
    chosen_rows = {}
    for stream in streams:
        row = front_runner(list(stream["margins"].values()))
        chosen_rows[stream["history_seconds"]] = row
        per_window[f"{stream['history_seconds']:g}"] = point_of(
            row, stream["history_seconds"])
    available = {history: row for history, row in chosen_rows.items() if row}
    if not available:
        return {"contract": contract, "rule": RULE, "per_window": per_window,
                "recommended": None, "deployed": None,
                "text": (f"No (window length, margin) point at {contract} "
                         "qualified: none kept the balanced accuracy on decided "
                         "frames above chance in every held-out story while also "
                         "beating the majority rate of its own decided subset. "
                         "The curve above is the answer, and no operating point "
                         "is recommended from it.")}
    best_history = max(available, key=lambda history: (
        available[history]["frames"]["balanced_accuracy_decided"],
        available[history]["margin"], -history))
    recommended = point_of(available[best_history], best_history)
    default_history, default_contract = deployed_window()
    deployed = None
    if default_history is not None and default_contract == contract:
        deployed = per_window.get(f"{default_history:g}")
    text = (
        f"Across every window length measured at {contract}, the rule picks "
        f"margin {recommended['margin']:.2f} at {recommended['history_seconds']:g}s "
        f"windows: coverage {recommended['coverage']:.3f}, accuracy "
        f"{_pct(recommended['accuracy_decided'])} and balanced accuracy "
        f"{_pct(recommended['balanced_accuracy_decided'])} on the decided frames "
        f"(per story {_pct(recommended['fold_balanced_min'])} to "
        f"{_pct(recommended['fold_balanced_max'])}, "
        f"{recommended['folds_above_chance']} above chance), against a majority "
        f"rate of {_pct(recommended['majority_decided'])} inside its own decided "
        "subset. Costs: per-class recall over all frames "
        f"{_pct(recommended['recall_over_all']['0'])}/"
        f"{_pct(recommended['recall_over_all']['1'])} (an abstention is a miss), "
        f"first decision after a median of "
        f"{_num(recommended['first_decision_seconds_median'], 1)}s, "
        f"{recommended['unavailable_frames']} frames unavailable rather than "
        f"uncertain, and {_num(recommended['false_selection_changes_per_minute'], 3)} "
        "false selection changes per minute."
    )
    if deployed is None:
        text += (f" No saved model runs at that window length, so nothing in "
                 f"`{DEPLOYED_MODEL}` can be operated at it yet.")
    elif deployed["history_seconds"] != recommended["history_seconds"]:
        text += (
            f" `{DEPLOYED_MODEL}` is a "
            f"{deployed['history_seconds']:g}s model, so what can be run today is "
            f"margin {deployed['margin']:.2f} at {deployed['history_seconds']:g}s: "
            f"coverage {deployed['coverage']:.3f}, accuracy "
            f"{_pct(deployed['accuracy_decided'])}, balanced accuracy "
            f"{_pct(deployed['balanced_accuracy_decided'])} on decided frames "
            f"(per story {_pct(deployed['fold_balanced_min'])} to "
            f"{_pct(deployed['fold_balanced_max'])}, "
            f"{deployed['folds_above_chance']} above chance), first decision "
            f"{_num(deployed['first_decision_seconds_median'], 1)}s, "
            f"{_num(deployed['false_selection_changes_per_minute'], 3)} false "
            f"selection changes per minute. The longer window buys "
            f"{_num(recommended['balanced_accuracy_decided'] - deployed['balanced_accuracy_decided'], 3)} "
            "balanced accuracy and "
            f"{_num(deployed['false_selection_changes_per_minute'] - recommended['false_selection_changes_per_minute'], 3)} "
            "fewer false changes per minute at "
            f"{_num(deployed['coverage'] - recommended['coverage'], 3)} coverage, "
            "and it costs a model fitted at that length.")
    else:
        text += (f" `{DEPLOYED_MODEL}` already runs at that window length, so this "
                 "point is runnable without refitting anything.")
    if deployed is not None:
        default_row = next(
            (stream["margins"][f"{MIN_MARGIN:g}"] for stream in streams
             if stream["history_seconds"] == deployed["history_seconds"]), None)
        if default_row is not None:
            text += (
                f" For comparison, the current default margin {MIN_MARGIN:g} at "
                f"{deployed['history_seconds']:g}s covers "
                f"{default_row['frames']['coverage']:.4f} of frames with "
                f"{default_row['frames']['decided']} decided frames in the whole "
                f"held-out corpus.")
    return {"contract": contract, "rule": RULE, "per_window": per_window,
            "recommended": recommended, "deployed_model": DEPLOYED_MODEL,
            "deployed": deployed, "text": text}


PLUMBING = """\
The margin is not a constant to edit; it is a run-policy choice, and the run
record must be able to state it (plan section 3.17 item 5, decision D-26).

1. `RunPolicy` (`src/nova2026/auditory/session.py`) gains
   `margin: float = MIN_MARGIN` (imported from `nova2026.auditory.config`), so an
   existing caller that passes no margin keeps `0.5` exactly, and `to_dict()`
   records the value under `"margin"`.
2. `AttentionSession.__init__` builds `AttentionController(margin=policy.margin,
   max_age=..., min_switch_windows=..., attenuation_db=...)` where it currently
   builds `AttentionController()`; the `controller=` argument keeps winning when
   one is supplied.
3. The demo CLI (`scripts/auditory_ui/demorun.py`, `demo.py`) gains `--margin`
   with default `None` meaning "whatever the policy default is", and the run
   record writes the effective value beside `check_channels`.
4. **Window length is not a policy field and cannot be one.** The session reads
   `window_seconds` from the decoder's contract, so a longer window needs a
   model fitted at that length. `models/auditory_kuleuven.npz` is a 5 s model;
   any 30 s/60 s operating point requires a new model, which is a separate
   deliverable and not this task's to write.
5. The proof that the default is unchanged is in
   `scripts/auditory/tests/test_margin_calibration.py`: `MIN_MARGIN` is still
   `0.5`, `AttentionController()` still carries `margin == 0.5`, and the
   calibrated constant is a different name that nothing defaults to.
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(RESULTS), help="where the reports go")
    parser.add_argument("--contracts", nargs="*", default=["64ch"],
                        choices=list(train_kuleuven.CONTRACTS))
    parser.add_argument("--histories", nargs="*", type=float, default=list(HISTORIES))
    parser.add_argument("--margins", nargs="*", type=float, default=list(MARGIN_LADDER))
    parser.add_argument("--subjects", nargs="*", help="restrict the whole run")
    parser.add_argument("--json", help="explicit output JSON path")
    parser.add_argument("--md", help="explicit output markdown path")
    args = parser.parse_args(argv)

    if MIN_MARGIN not in args.margins:
        raise ValueError(
            f"The ladder must contain the value it judges: {MIN_MARGIN:g} is missing."
        )
    started = time.time()
    key, _ = feature_cache.cache_key()
    paths = sorted(feature_cache.cache_directory(key).glob("S*/trial_*.npz"))
    corpus = train_kuleuven.Corpus(key, paths).select(args.subjects)
    document = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": "python -B -m scripts.auditory.margin_calibration "
                   f"--out {args.out} --contracts {' '.join(args.contracts)} "
                   f"--histories {' '.join(f'{h:g}' for h in args.histories)}",
        "python": platform.python_version(),
        "cache_key": key,
        "cache_directory": str(feature_cache.cache_directory(key)),
        "cache_trials": len(corpus.trials),
        "subjects": corpus.subjects,
        "groups": corpus.groups,
        "folds": corpus.groups,
        "fold_definition": "train_kuleuven.held_out_story_folds over "
                           "kuleuven_contract.group_key (rep_* folded into its base)",
        "alpha": ALPHA,
        "min_margin_default": MIN_MARGIN,
        "margin_ladder": list(args.margins),
        "histories": list(args.histories),
        "contracts": list(args.contracts),
        "frame_seconds": FRAME_SECONDS,
        "warmup_seconds": WARMUP_SECONDS,
        "controller": {"min_switch_windows": MIN_SWITCH_WINDOWS, "max_age": MAX_AGE,
                       "attenuation_db": ATTENUATION_DB},
        "nulls": {
            "majority_trial_rate": 256 / 320,
            "majority_time_rate": 48894.0 / (48894.0 + 25087.5),
            "note": "The majority candidate is A: 0.80 of the trials and 0.6609 of "
                    "the labeled time, so raw accuracy is reported beside the "
                    "majority of its own decided subset and beside the balanced "
                    "accuracy, whose chance line is 0.50.",
        },
        "streams": [],
    }
    for contract in args.contracts:
        for history in args.histories:
            print(f"[{time.time() - started:6.1f}s] scoring {contract} {history:g}s",
                  flush=True)
            stream = stream_of(corpus, contract, history,
                               progress=_progress(started, contract, history))
            print(f"[{time.time() - started:6.1f}s] sweeping {len(args.margins)} "
                  f"margins on {len(stream['trials'])} held-out trials", flush=True)
            stream["margins"] = sweep_margins(stream, args.margins)
            # The per-window score streams are what the sweep consumed; keeping
            # them would make the JSON a copy of the cache. What is kept is the
            # composition of the held-out side, which is what makes the numbers
            # checkable.
            stream["trials"] = [
                {"fold": trial["fold"], "subject": trial["subject"],
                 "trial_id": trial["trial_id"], "group": trial["group"],
                 "label": trial["label"], "windows": int(len(trial["arrivals"])),
                 "frames": int(len(trial["frame_times"]))}
                for trial in stream["trials"]
            ]
            document["streams"].append(stream)
    document["fold_trial_counts"] = sorted(
        {count for stream in document["streams"]
         for count in stream["fold_sizes"].values()}
    ) or [0]
    document["recommendation"] = recommend(document, args.contracts[0])
    document["recommendation"]["plumbing"] = PLUMBING
    document["elapsed_seconds"] = round(time.time() - started, 1)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = Path(args.json) if args.json else Path(args.out) / \
        f"aad_margin_calibration_{stamp}.json"
    md_path = Path(args.md) if args.md else Path(args.out) / \
        f"aad_margin_calibration_{stamp}.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    write_report(document, md_path)
    print(f"[{time.time() - started:6.1f}s] wrote {json_path} and {md_path}", flush=True)
    return 0


def _progress(started, contract, history):
    def report(index, trial):
        if index % 40 == 0:
            print(f"[{time.time() - started:6.1f}s] {contract} {history:g}s "
                  f"{index} trials scored", flush=True)
    return report


if __name__ == "__main__":
    raise SystemExit(main())
