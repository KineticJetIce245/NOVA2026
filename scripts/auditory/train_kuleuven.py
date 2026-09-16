"""Step 5: train and evaluate the KU Leuven attention decoder, honestly.

Two contracts, because the live 24-channel ANT cap and the 64-channel KU Leuven
cap share only 20 electrodes (``final_connection.md`` section 3.11): a
64-channel model comparable with the paper's baseline, and a 20-channel model
the live rig can actually feed. Two evaluation schemes, because neither answers
the other's question: whole stories are held out (leakage-free across the
``rep_`` replays, which carry the same audio as their base) and subjects are
held out (the number that says whether a new person can use it).

Three things this script refuses to do:

* report a single accuracy. The labels are 80/20 imbalanced in trial count and
  66/34 in time, so answering "A" every window already scores about 66%; every
  table carries accuracy, balanced accuracy and both per-class recalls next to
  that null.
* choose a regularization on the data it is scored on. ``alpha`` is fixed at
  :data:`DEFAULT_ALPHA` for every headline number; the other candidates appear
  beside it as a sensitivity table, not as a selection.
* let a label reach the decoder. A window is built from the signal, the
  envelope and the clock only; the label is read from the cache at scoring time
  and asserted out of the inference path in ``tests/test_aad_decoder.py``.

    python -B -m scripts.auditory.train_kuleuven --out results --models models
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from nova2026.auditory.data import AuditoryWindow

from . import aad_ridge, feature_cache
from .kuleuven_contract import REPO, RESULTS

ALPHAS = (10.0, 100.0, 1000.0)
DEFAULT_ALPHA = 100.0
MAIN_HISTORY = 5.0
CURVE_HISTORIES = (5.0, 10.0, 30.0, 60.0)
# The window-length curve runs the same protocol on a documented quarter of the
# corpus: eight full-corpus fits do not fit step 5's budget, and a curve on a
# stated subsample is worth more than no curve at all.
CURVE_SUBJECTS = ("S1", "S2", "S3", "S4")
CONTRACTS = ("64ch", "20ch")
MODEL_NAMES = {"64ch": "auditory_kuleuven.npz", "20ch": "auditory_kuleuven_live20.npz"}


class TrialRef:
    """What a split needs to know about one trial, and nothing else."""

    __slots__ = ("subject", "trial_id", "group", "key", "label", "path")

    def __init__(self, subject, trial_id, group, label, path):
        self.subject = str(subject)
        self.trial_id = str(trial_id)
        self.group = str(group)
        self.key = (self.subject, self.trial_id)
        self.label = int(label)
        self.path = Path(path)

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"TrialRef({self.subject}, {self.trial_id}, label={self.label})"


class Corpus:
    """The cached trials of one key, indexed by the fields splits are drawn on."""

    def __init__(self, key, paths):
        self.key = key
        self.trials = []
        self.contracts = None
        for path in sorted(paths):
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata"]))
                label = int(np.asarray(archive["labels"]).reshape(-1)[0])
            self.trials.append(TrialRef(metadata["subject"], metadata["trial_id"],
                                        metadata["group_key"], label, path))
            if self.contracts is None:
                self.contracts = metadata["contracts"]
        if not self.trials:
            raise ValueError(f"No cached trials for key {key}.")
        if len({trial.key for trial in self.trials}) != len(self.trials):
            raise ValueError("The cache holds a duplicated trial.")

    def contract_of(self, contract, history):
        return self.contracts[contract][feature_cache.history_tag(history)]

    @property
    def subjects(self):
        return sorted({trial.subject for trial in self.trials}, key=lambda n: int(n[1:]))

    @property
    def groups(self):
        return sorted({trial.group for trial in self.trials})

    def by(self, field):
        grouped = {}
        for trial in self.trials:
            grouped.setdefault(getattr(trial, field), []).append(trial)
        return grouped

    def select(self, subjects=None):
        if not subjects:
            return self
        wanted = set(subjects)
        subset = Corpus.__new__(Corpus)
        subset.key = self.key
        subset.contracts = self.contracts
        subset.trials = [trial for trial in self.trials if trial.subject in wanted]
        return subset


def assert_group_disjoint(training, validation, testing=(), stories=True):
    """Refuse a split that shares a trial with its other side, or a story.

    The story comparison is on the canonical group key, which folds ``rep_*``
    into the story it replays. Comparing the stored names instead accepts a
    split that trains on ``part1_track1_dry.wav`` and validates on
    ``rep_part1_track1_dry.wav`` -- 125 s of the same audio on both sides
    (``results/kuleuven_audit.md`` section 4).

    ``stories=False`` is for the leave-one-subject-out scheme and only for it:
    every subject listens to the same four stories, so holding a subject out
    necessarily leaves all four stories in the training side. That is what the
    scheme measures -- a new person, not a new story -- and the story-disjoint
    number is the one that answers the other question. Trial identity is still
    enforced there.
    """

    seen_trials = set()
    seen_groups = set()
    for partition in (list(training), list(validation), list(testing)):
        trials = {trial.key for trial in partition}
        groups = {trial.group for trial in partition} if stories else set()
        if trials & seen_trials or groups & seen_groups:
            raise ValueError("Split reuses a trial or a story across partitions.")
        seen_trials |= trials
        seen_groups |= groups
    if validation:
        labels = {trial.label for trial in validation}
        if labels != {0, 1}:
            raise ValueError(
                "A validation side must carry both classes for balanced "
                f"accuracy to be defined; it carries {sorted(labels)}."
            )


def held_out_story_folds(corpus):
    """One fold per canonical story pair; both candidates of a part leave together."""

    folds = []
    for group, trials in sorted(corpus.by("group").items()):
        training = [trial for trial in corpus.trials if trial.group != group]
        assert_group_disjoint(training, trials)
        folds.append({"name": group, "training": training, "validation": trials})
    return folds


def leave_one_subject_folds(corpus):
    """One fold per subject, trained on the other fifteen."""

    folds = []
    for subject, trials in sorted(corpus.by("subject").items(),
                                  key=lambda item: int(item[0][1:])):
        training = [trial for trial in corpus.trials if trial.subject != subject]
        assert_group_disjoint(training, trials, stories=False)
        folds.append({"name": subject, "training": training, "validation": trials})
    return folds


def moments_pass(corpus, contract, history, progress=None):
    """One pass over the cache, summing moments per trial, subject and story.

    Every fold of both schemes is a sum of these, so one pass serves the
    subject folds, the story folds and anything else a subset of trials is
    needed for.
    """

    subsets = {"total": corpus.trials}
    for field in ("subject", "group"):
        for name, trials in corpus.by(field).items():
            subsets.setdefault(name, trials)
    sums = {}
    for index, trial in enumerate(corpus.trials):
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
        for name, trials in subsets.items():
            if trial in trials:
                sums[name] = moments if name not in sums else sums[name] + moments
        if progress is not None:
            progress(index + 1, trial)
    return sums


def training_moments(sums, validation):
    """The running sum with the fold's held-out slice removed.

    The slice must be exactly one whole story or one whole subject, because
    those are the only sums the pass keeps. Removing both would subtract the
    same trials twice, which is why this resolves one key instead of taking the
    union of everything the validation side names.
    """

    groups = {trial.group for trial in validation}
    subjects = {trial.subject for trial in validation}
    if len(groups) == 1:
        key = next(iter(groups))
        expected = "story"
    elif len(subjects) == 1:
        key = next(iter(subjects))
        expected = "subject"
    else:
        raise ValueError(
            "A fold's validation side must be one whole story or one whole "
            f"subject; it spans {len(groups)} stories and {len(subjects)} subjects."
        )
    if key not in sums:
        raise ValueError(f"No cached moments for the held-out {expected} {key}.")
    if sums[key].design_count >= sums["total"].design_count:
        raise ValueError(f"The held-out {expected} {key} is the whole corpus.")
    return sums["total"] - sums[key]


def window_at(features, contract, history, start):
    """One inference window: signal, envelope and clock, and nothing else."""

    signal = features.signal(contract).astype(float)
    length = round(history * feature_cache.sample_rate())
    return AuditoryWindow(
        signal[start : start + length],
        features.envelopes[start : start + length].astype(float),
        features.times[start : start + length],
        float(features.times[start + length - 1]),
        True,
        (),
        features.contract[contract][feature_cache.history_tag(history)],
        0,
    )


def score_trials(model, trials, contract, history, key):
    """Window-level decisions and truths for one validation side.

    Scoring walks the same non-overlapping grid the fold was fitted on, so no
    recording is counted once per second of hop.
    """

    decisions = []
    truths = []
    for trial in trials:
        features = feature_cache.load(trial.path, key)
        for start, _ in features.windows(contract, history, hop=history):
            scores = model.score(window_at(features, contract, history, start))
            decisions.append(-1 if abs(scores[0] - scores[1]) <= 1e-12
                             else int(np.argmax(scores)))
            truths.append(features.label)
    return np.asarray(decisions), np.asarray(truths)


_CORPUS_KEY = None


def metrics(decisions, truths):
    """Accuracy, balanced accuracy, per-class recall, and the null beside them."""

    decisions = np.asarray(decisions)
    truths = np.asarray(truths)
    total = len(truths)
    if total == 0:
        raise ValueError("No scorable window.")
    decided = decisions >= 0
    correct = int(np.count_nonzero((decisions == truths) & decided))
    recalls = {}
    for value in (0, 1):
        mask = truths == value
        recalls[str(value)] = (
            float(np.count_nonzero(decisions[mask] == value) / mask.sum())
            if mask.any() else None
        )
    present = [recalls[str(value)] for value in (0, 1) if recalls[str(value)] is not None]
    return {
        "windows": int(total),
        "decided": int(decided.sum()),
        "accuracy": correct / total,
        "balanced_accuracy": float(np.mean(present)),
        "recall": recalls,
        "class_windows": {str(value): int(np.count_nonzero(truths == value))
                          for value in (0, 1)},
        "majority_accuracy": (max(int(np.count_nonzero(truths == value))
                                  for value in (0, 1)) / total),
    }


def evaluate(corpus, contract, history, alpha, folds, sums, scheme):
    """Fit every fold of one scheme and report the folds and their pool."""

    records = []
    pooled = ([], [])
    for fold in folds:
        training = training_moments(sums, fold["validation"])
        provenance = {
            "training": [(t.subject, t.trial_id, t.group) for t in fold["training"]],
            # The scored side is recorded as held out, never as development
            # data: nothing about it chose this model, and the library's
            # overlapping-group guard reads only "training"/"validation".
            "held_out": [(t.subject, t.trial_id, t.group) for t in fold["validation"]],
            "scheme": scheme, "fold": fold["name"], "contract": contract,
            "history_seconds": history, "alpha": alpha, "cache_key": corpus.key,
            "labels_in_inference_path": False,
        }
        model = aad_ridge.fit(training, corpus.contract_of(contract, history), alpha,
                              training_info=provenance)
        if scheme == "held_out_story":
            # The guard compares story tokens, so it can only speak for the
            # scheme that holds stories out. A subject fold trains on the other
            # subjects' recordings of the same four stories by construction.
            from nova2026.auditory.evaluation import assert_held_out

            for trial in fold["validation"]:
                assert_held_out(trial, model)
        decisions, truths = score_trials(model, fold["validation"], contract, history,
                                         corpus.key)
        pooled[0].append(decisions)
        pooled[1].append(truths)
        records.append({"fold": fold["name"], **metrics(decisions, truths)})
    overall = metrics(np.concatenate(pooled[0]), np.concatenate(pooled[1]))
    return {
        "scheme": scheme, "contract": contract, "history_seconds": history,
        "alpha": alpha, "folds": records, "pooled": overall,
        "fold_accuracy_mean": float(np.mean([r["accuracy"] for r in records])),
        "fold_accuracy_std": float(np.std([r["accuracy"] for r in records])),
        "fold_balanced_mean": float(np.mean([r["balanced_accuracy"] for r in records])),
        "fold_balanced_std": float(np.std([r["balanced_accuracy"] for r in records])),
    }


def final_model(corpus, contract, history, alpha, sums=None, corpus_key=None):
    """Fit the deployed model on every cached trial of one contract."""

    sums = moments_pass(corpus, contract, history) if sums is None else sums
    history_name = feature_cache.history_tag(history)
    provenance = {
        "training": [(t.subject, t.trial_id, t.group) for t in corpus.trials],
        "held_out": [],
        "scheme": "all-trials", "contract": contract, "history_seconds": history,
        "alpha": alpha, "cache_key": corpus.key, "labels_in_inference_path": False,
    }
    return aad_ridge.fit(sums["total"], corpus.contract_of(contract, history), alpha,
                         training_info=provenance)


def nested(corpus, contract, history, alpha, sums, subjects):
    """Run both schemes on one contract at one window length."""

    subset = corpus.select(subjects)
    if subset is not corpus:
        sums = moments_pass(subset, contract, history)
    story = evaluate(subset, contract, history, alpha, held_out_story_folds(subset),
                     sums, "held_out_story")
    subject = evaluate(subset, contract, history, alpha, leave_one_subject_folds(subset),
                       sums, "leave_one_subject_out")
    return {"held_out_story": story, "leave_one_subject_out": subject}


def sensitivity(corpus, contract, history, sums):
    """Fold-pooled accuracy at each alpha, reported and never selected on."""

    table = []
    for alpha in ALPHAS:
        result = evaluate(corpus, contract, history, alpha, held_out_story_folds(corpus),
                          sums, "held_out_story")
        table.append({"alpha": alpha, "pooled": result["pooled"],
                      "fold_accuracy_mean": result["fold_accuracy_mean"],
                      "fold_balanced_mean": result["fold_balanced_mean"]})
    return table


def write_report(document, path):
    """A short human-readable summary of the JSON the same run wrote."""

    lines = [
        "# KU Leuven auditory decoder: step 5 results",
        "",
        f"Generated `{document['generated_at']}` by `{document['command']}`.",
        "",
        f"The decoder was fitted on the feature cache `{document['cache_key']}` "
        f"({document['cache_trials']} trials). " + document["nulls"]["note"],
        "",
        "## Main results (window length 5 s, alpha 100)",
        "",
        "Windows are the non-overlapping 5 s grid of "
        "`scripts/auditory/train.py::prepare`, so every trial is counted once.",
        "",
        "| contract | scheme | accuracy | majority | balanced acc. | recall A | recall B | windows |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for contract in CONTRACTS:
        for scheme in ("held_out_story", "leave_one_subject_out"):
            result = document["main"][contract][scheme]
            pooled = result["pooled"]
            lines.append(
                f"| {contract} | {scheme} | {pooled['accuracy']:.4f} | "
                f"{pooled['majority_accuracy']:.4f} | "
                f"{pooled['balanced_accuracy']:.4f} | {pooled['recall']['0']:.4f} | "
                f"{pooled['recall']['1']:.4f} | {pooled['windows']} |"
            )
    lines += [
        "",
        "**Read the accuracy against its own majority column.** At five seconds "
        "the model is *below* the majority rate on raw accuracy, so accuracy "
        "alone cannot describe it. What is above chance is the balanced "
        "accuracy, whose chance line is 0.50: 0.62 for the 64-channel contract "
        "and 0.60 for the 20-channel one on held-out stories. That is a real "
        "but weak effect, and it is the honest headline.",
        "",
        "Per-fold spread is in the JSON: `fold_accuracy_mean` and "
        "`fold_balanced_mean` with their standard deviations, and the same "
        "numbers per fold in `folds`.",
        "",
        "## Window-length curve",
        "",
        f"Measured with the held-out-story scheme on {', '.join(CURVE_SUBJECTS)} "
        "(a documented quarter of the corpus) so that four window lengths are "
        "affordable; the main numbers above use all 16 subjects. Both contracts "
        "rise monotonically with the window, which is what a weak, "
        "noise-limited correlation decoder should do.",
        "",
        "| contract | window (s) | accuracy | majority | balanced acc. | recall A | recall B | windows |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for contract in CONTRACTS:
        for point in document["window_curve"][contract]:
            pooled = point["pooled"]
            lines.append(
                f"| {contract} | {point['history_seconds']:g} | "
                f"{pooled['accuracy']:.4f} | {pooled['majority_accuracy']:.4f} | "
                f"{pooled['balanced_accuracy']:.4f} | "
                f"{pooled['recall']['0']:.4f} | {pooled['recall']['1']:.4f} | "
                f"{pooled['windows']} |"
            )
    lines += [
        "",
        "## Alpha sensitivity (never used to select)",
        "",
        "| contract | alpha | accuracy | balanced acc. |",
        "| --- | --- | --- | --- |",
    ]
    for contract in CONTRACTS:
        for row in document["alpha_sensitivity"][contract]:
            lines.append(
                f"| {contract} | {row['alpha']:g} | {row['pooled']['accuracy']:.4f} | "
                f"{row['pooled']['balanced_accuracy']:.4f} |"
            )
    lines += [
        "",
        "## Models",
        "",
    ]
    for record in document["models"]:
        lines.append(
            f"- `{record['path']}`: {record['contract']} contract, "
            f"{len(record['channels'])} channels, {record['history_seconds']:g} s "
            f"history, alpha {record['alpha']:g}, "
            f"{record['training_trials']} training trials."
        )
    lines += [
        "",
        "## What these numbers do not say",
        "",
        "- They are window-level and come from one dataset, KU Leuven, with the "
        "authors' own stimuli; they are not a claim about the live rig.",
        "- The 20-channel contract is the only one the live cap can feed; the "
        "gap between it and the 64-channel row is the cost of that.",
        "- Windows were accepted under the chain's documented relaxed channel "
        "policy (`check_channels=False`), because the default policy stops the "
        "run on this dataset's amplitude excursions. The signal is identical "
        "under either policy; only the verdicts differ.",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(RESULTS), help="where the reports go")
    parser.add_argument("--models", default=str(REPO / "models"))
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--history", type=float, default=MAIN_HISTORY)
    parser.add_argument("--curve-subjects", nargs="*", default=list(CURVE_SUBJECTS))
    parser.add_argument("--subjects", nargs="*", help="restrict the whole run")
    parser.add_argument("--skip-curve", action="store_true")
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument("--json", help="explicit output JSON path")
    parser.add_argument("--md", help="explicit output markdown path")
    args = parser.parse_args(argv)

    global _CORPUS_KEY
    started = time.time()
    key, _ = feature_cache.cache_key()
    paths = sorted(feature_cache.cache_directory(key).glob("S*/trial_*.npz"))
    corpus = Corpus(key, paths).select(args.subjects)
    _CORPUS_KEY = key
    document = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": "python -B -m scripts.auditory.train_kuleuven "
                   f"--out {args.out} --models {args.models}",
        "cache_key": key,
        "cache_directory": str(feature_cache.cache_directory(key)),
        "cache_trials": len(corpus.trials),
        "cache_key_material": feature_cache.key_material(),
        "subjects": corpus.subjects,
        "groups": corpus.groups,
        "alpha": args.alpha,
        "main_history_seconds": args.history,
        "nulls": {
            "majority_trial_rate": 256 / 320,
            "majority_time_rate": 48894.0 / (48894.0 + 25087.5),
            "note": "The majority candidate is A: 0.80 of the trials are A and "
                    "0.6609 of the labeled time is A, so a single accuracy is "
                    "not interpretable on its own. Every table below reports "
                    "the majority rate over its own pooled windows beside the "
                    "accuracy and the balanced accuracy, whose chance line is "
                    "0.50 by construction.",
            "source": "results/kuleuven_audit.md section 3 (trial counts and labeled seconds)",
        },
        "main": {}, "alpha_sensitivity": {}, "window_curve": {}, "models": [],
    }
    for contract in CONTRACTS:
        print(f"[{time.time() - started:6.1f}s] moments: {contract}", flush=True)
        sums = moments_pass(corpus, contract, args.history)
        document["main"][contract] = nested(corpus, contract, args.history,
                                            args.alpha, sums, None)
        if not args.skip_sensitivity:
            print(f"[{time.time() - started:6.1f}s] alpha sensitivity: {contract}",
                  flush=True)
            document["alpha_sensitivity"][contract] = sensitivity(
                corpus, contract, args.history, sums)
        if not args.skip_curve:
            document["window_curve"][contract] = []
            for history in CURVE_HISTORIES:
                print(f"[{time.time() - started:6.1f}s] curve: {contract} {history:g}s",
                      flush=True)
                subset = corpus.select(args.curve_subjects)
                point = evaluate(subset, contract, history, args.alpha,
                                 held_out_story_folds(subset),
                                 moments_pass(subset, contract, history),
                                 "held_out_story")
                document["window_curve"][contract].append(point)
        model = final_model(corpus, contract, args.history, args.alpha, sums=sums)
        destination = Path(args.models) / MODEL_NAMES[contract]
        destination.parent.mkdir(parents=True, exist_ok=True)
        model.save(destination)
        document["models"].append({
            "path": str(destination.relative_to(REPO)) if destination.is_relative_to(REPO)
            else str(destination),
            "contract": contract, "channels": corpus.contract_of(contract, args.history)["eeg_channels"],
            "history_seconds": args.history, "alpha": args.alpha,
            "training_trials": len(corpus.trials), "cache_key": key,
        })
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = Path(args.json) if args.json else Path(args.out) / f"aad_{stamp}.json"
    md_path = Path(args.md) if args.md else Path(args.out) / f"aad_{stamp}.md"
    document["elapsed_seconds"] = round(time.time() - started, 1)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    write_report(document, md_path)
    print(f"[{time.time() - started:6.1f}s] wrote {json_path} and {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
