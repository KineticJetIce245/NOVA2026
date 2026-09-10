"""Compare controllers and fault conditions with a fixed held-out trial."""

import argparse
import json
from pathlib import Path

from nova2026.auditory.data import load_trial
from nova2026.auditory.decoder import RidgeDecoder
from nova2026.auditory.evaluation import inject_fault

from .replay import replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    trial = load_trial(args.trial)
    model = RidgeDecoder.load(args.model)
    for partition in ("training", "validation"):
        for subject, identity, group in model.training_info.get(partition, []):
            if (subject, identity) == (
                trial.subject,
                trial.trial_id,
            ) or group == trial.group:
                raise ValueError("Evaluation overlaps model development data.")
    results = []
    for fault in ("none", "dropout", "artifact", "mismatched"):
        recording = inject_fault(trial, fault)
        for mode in ("quality", "hysteresis", "neutral", "oracle"):
            for margin in (0.015, 0.03, 0.06):
                _, _, metrics = replay(recording, model, margin=margin, mode=mode)
                results.append({"fault": fault, "margin": margin, **metrics})
    for delay in (1.0, 5.0):
        _, _, metrics = replay(trial, model, delay=delay)
        results.append({"fault": "delayed_inference", "delay": delay, **metrics})
    for offset in (-0.25, 0.25):
        _, _, metrics = replay(trial, model, audio_offset=offset)
        results.append({"fault": "audio_clock_offset", "offset": offset, **metrics})
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {len(results)} comparisons to {path}")


if __name__ == "__main__":
    main()
