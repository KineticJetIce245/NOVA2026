"""Train a ridge decoder with a separate, explicitly grouped validation split."""

import argparse
import json
from pathlib import Path

import numpy as np

from nova2026.auditory.config import AuditoryConfig
from nova2026.auditory.data import load_trial
from nova2026.auditory.decoder import RidgeDecoder
from nova2026.auditory.evaluation import check_split

from .runner import labels_for_window, replay_windows


def prepare(trials, config, history, **replay_options):
    examples = []
    for trial in trials:
        next_start = -np.inf
        for window in replay_windows(trial, config, history, 1.0, **replay_options):
            if window.valid and window.timestamps[0] >= next_start:
                examples.append((window, labels_for_window(trial, window)))
                next_start = window.timestamps[0] + history
    return examples


def train(training, validation, history=5.0, alphas=(10.0, 100.0, 1000.0), base=None, *, false_fire_rate=.01, allow_flat_channels=(), **replay_options):
    check_split(training, validation)
    arms = {getattr(t, 'presentation', None) for t in training+validation}
    if len(arms-{None}) > 1:
        raise ValueError('Train separate calibration models for dichotic and diotic recordings.')
    config = AuditoryConfig()
    # Training and inference use the same history and hop contract. We use
    # nonoverlapping windows initially to avoid counting the same EEG repeatedly.
    training_examples = prepare(training, config, history, **replay_options)
    validation_examples = prepare(validation, config, history, **replay_options)
    best_model = None
    best_accuracy = -1.0
    report = []
    for alpha in alphas:
        model = RidgeDecoder(config, alpha)
        info = {
            "training": [(t.subject, t.trial_id, t.group) for t in training],
            "validation": [(t.subject, t.trial_id, t.group) for t in validation],
            "history": history,
            "step": 1.0,
            "presentation": next(iter(arms)) if len(arms) == 1 else None,
        }
        if base is not None:
            from nova2026.auditory.evaluation import assert_held_out
            for trial in validation:
                assert_held_out(trial, base)
            info["training"] += base.training_info.get("training", [])
            info["training"] += base.training_info.get("validation", [])
        model.fit(training_examples, info, base=base, allow_flat_channels=allow_flat_channels)
        correct = 0
        total = 0
        for window, labels in validation_examples:
            # Mixed/unknown windows cannot provide a single classification target.
            unique = np.unique(labels)
            if len(unique) != 1 or unique[0] not in (0, 1):
                continue
            scores = model.score(window)
            if abs(scores[0] - scores[1]) > 1e-12:
                correct += int(np.argmax(scores) == unique[0])
            total += 1
        if total == 0:
            raise ValueError("No steady, labeled validation windows.")
        accuracy = correct / total
        report.append({"alpha": alpha, "accuracy": accuracy, "windows": total})
        if accuracy > best_accuracy:
            best_model = model
            best_accuracy = accuracy
    if best_model is None:
        raise ValueError("At least one regularization candidate is required.")
    from nova2026.auditory.gate import DecisionGate
    from nova2026.auditory.evaluation import inject_fault
    null_examples = prepare([inject_fault(t, "mismatched") for t in training],
                            config, history, **replay_options)
    differences = [float(np.diff(best_model.score(w))[0] * -1) for w, _ in null_examples]
    if len(differences) < 20:
        # Short diagnostic fixtures can still fit reconstruction weights, but
        # are not a calibrated participant model. UI and play fail closed on
        # the absent gate. Never pretend four windows establish a 1% tail.
        best_model.training_info['gate_unavailable'] = 'Fewer than 20 null windows; collect more calibration.'
        return best_model, report
    gate = DecisionGate.fit(differences, false_fire_rate)
    best_model.training_info['decision_gate'] = gate.to_dict()
    best_model.training_info['replay_options'] = dict(replay_options)
    scores, truth = [], []
    for window, labels in validation_examples:
        unique = np.unique(labels)
        if len(unique) == 1 and unique[0] in (0, 1):
            scores.append(best_model.score(window))
            truth.append(int(unique[0]))
    best_model.training_info['gate_validation'] = gate.metrics(scores, truth)
    best_model.training_info['gate_validation']['interpretation'] = (
        'Development validation used for alpha selection; not independent test accuracy. '
        'False-fire rate is a calibration target, not a measured population guarantee.')
    report.append({'gate': gate.to_dict(), **best_model.training_info['gate_validation']})
    return best_model, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--validation", nargs="+", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--history", "--window", type=float, default=5.0)
    parser.add_argument("--base", help="Optional .npz decoder prior with matching contract")
    parser.add_argument("--timing-profile", help="Measured audio profile for calibration recorded through that exact path")
    parser.add_argument('--audio-offset', type=float, default=None, help='Override per-trial audio onset offset in seconds')
    parser.add_argument('--no-channel-check', action='store_true')
    parser.add_argument('--max-bad-channels', type=int, default=0)
    parser.add_argument('--exclude-channels', default='')
    parser.add_argument('--false-fire-rate', type=float, default=.01)
    parser.add_argument('--allow-flat-channels', default='', help='Explicitly disable named flat training electrodes; preserves montage')
    args = parser.parse_args()
    training = [load_trial(path) for path in args.train]
    validation = [load_trial(path) for path in args.validation]
    base = RidgeDecoder.load(args.base) if args.base else None
    model, report = train(training, validation, args.history, base=base,
                          audio_offset=args.audio_offset, check_channels=not args.no_channel_check,
                          max_bad_channels=args.max_bad_channels,
                          exclude_channels=tuple(c.strip() for c in args.exclude_channels.split(',') if c.strip()),
                          false_fire_rate=args.false_fire_rate,
                          allow_flat_channels=tuple(c.strip() for c in args.allow_flat_channels.split(',') if c.strip()))
    if args.timing_profile:
        from nova2026.auditory.timing import validate_audio_profile
        profile = json.loads(Path(args.timing_profile).read_text())
        for trial in training + validation:
            validate_audio_profile(profile, trial.audio_rate, max(1, round(trial.audio_rate * .032)))
        model.training_info["audio_timing_profile"] = profile
    path = Path(args.model)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    path.with_suffix(".validation.json").write_text(json.dumps(report, indent=2))
    print(f"Saved decoder: {path}")


if __name__ == "__main__":
    main()
