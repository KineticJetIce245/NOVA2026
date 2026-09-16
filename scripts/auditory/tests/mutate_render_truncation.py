"""Show that the truncation tests fail under a named mutation, then restore.

Plan section 6.4 requires it: a test that passes is not yet evidence that it
tests anything. Each mutation below breaks the *shipped* behaviour - the exact
behaviour the change under test introduced - the named test is run against it,
and the test must go red. The file is restored in a ``finally`` block and its
bytes are compared before and after, so a run that dies half way cannot leave a
mutated tree behind silently.

The defect this locks was a call site, not a library function: ``render_stereo``
was always honest about the array it was handed. So the mutations are applied to
the call sites as well as to the rule, and each one names the test it kills.

    .venv\\Scripts\\python.exe -B scripts\\auditory\\tests\\mutate_render_truncation.py

Exit code 0 only when every mutation was caught by the test it names.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SUITE = "scripts.auditory.tests.test_render_truncation"

MUTATIONS = (
    {
        "name": "pass the untruncated candidates at the demo call site",
        "module": "scripts/auditory_ui/demo.py",
        "find": "    report = render_stereo(\n        candidates,\n        out,",
        "replace": "    report = render_stereo(\n        trial.audio,\n        out,",
        "test": f"{SUITE}.CallSiteTests.test_the_demo_call_site_renders_the_eeg_span",
        "note": "the code as it stood before this change: the 394.00 s file for a 389 s trial",
    },
    {
        "name": "pass the untruncated candidates at the session call site",
        "module": "scripts/auditory_ui/session.py",
        "find": "    report = render_stereo(\n        candidates,\n        args.media_out,",
        "replace": "    report = render_stereo(\n        trial.audio,\n        args.media_out,",
        "test": f"{SUITE}.CallSiteTests.test_the_session_call_site_renders_the_eeg_span",
        "note": "the session path, whose docstring claimed the cut while the code did not make it",
    },
    {
        "name": "pass the untruncated candidates in the perturbation matrix",
        "module": "scripts/auditory_ui/demorun.py",
        "find": "        candidates, out, sample_rate=round(trial.audio_rate), **kwargs",
        "replace": "        trial.audio, out, sample_rate=round(trial.audio_rate), **kwargs",
        "test": f"{SUITE}.CallSiteTests.test_the_demorun_call_site_renders_the_eeg_span",
        "note": "all four demorun cases back on a file longer than the session",
    },
    {
        "name": "count candidates from the EEG's sample count, not its last sample",
        "module": "src/nova2026/auditory/render.py",
        "find": "    return int(np.floor((int(eeg_samples) - 1) * audio_rate / eeg_rate)) + 1",
        "replace": "    return int(np.floor(int(eeg_samples) * audio_rate / eeg_rate)) + 1",
        "test": f"{SUITE}.TruncationRuleTests.test_the_restated_rule_agrees_with_the_frozen_contract",
        "note": "an off-by-one that adds one EEG sample of audio back to the file",
    },
    {
        "name": "pad a short candidate set out to the EEG prefix",
        "module": "src/nova2026/auditory/render.py",
        "find": "    keep = min(eeg_prefix_samples(eeg_samples, eeg_rate, audio_rate), len(values))",
        "replace": (
            "    keep = eeg_prefix_samples(eeg_samples, eeg_rate, audio_rate)\n"
            "    if keep > len(values):\n"
            "        values = np.concatenate(\n"
            "            (values, np.zeros((keep - len(values), values.shape[1]))), axis=0\n"
            "        )"
        ),
        "test": f"{SUITE}.TruncationArithmeticTests.test_shorter_candidates_are_left_alone",
        "note": "silence appended to a trial that was legitimately shorter than its EEG",
    },
)


def digest(path: Path) -> str:
    """SHA256 of one file, so a mutation cannot be left behind unnoticed."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_test(test: str) -> tuple[int, str]:
    """Run one test and return its exit code and a one-line tail of its output."""

    completed = subprocess.run(
        [sys.executable, "-B", "-m", "unittest", test],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    tail = [line for line in (completed.stdout + completed.stderr).splitlines() if line.strip()]
    return completed.returncode, " | ".join(tail[-3:])[:400]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/render_truncation_mutation.json")
    args = parser.parse_args(argv)

    baseline_code, baseline_tail = run_test(SUITE)
    records = []
    if baseline_code != 0:
        print(f"the unmutated suite is already red; nothing to prove: {baseline_tail}")
        return 1
    print(f"baseline: {SUITE} is green")

    for mutation in MUTATIONS:
        path = REPO / mutation["module"]
        before = digest(path)
        # ``newline=""`` on both sides: a text-mode round trip would translate
        # every LF into CRLF, which is a whole-file diff and a false alarm.
        source = path.read_text(encoding="utf-8", newline="")
        if mutation["find"] not in source:
            records.append(
                {
                    "mutation": mutation["name"],
                    "test": mutation["test"],
                    "caught": False,
                    "detail": "the mutation target was not found; the code moved",
                }
            )
            continue
        path.write_text(
            source.replace(mutation["find"], mutation["replace"], 1),
            encoding="utf-8",
            newline="",
        )
        try:
            code, tail = run_test(mutation["test"])
        finally:
            path.write_text(source, encoding="utf-8", newline="")
            restored = digest(path) == before
        if not restored:
            print(f"FATAL: {mutation['module']} was not restored")
            return 2
        caught = code != 0
        records.append(
            {
                "mutation": mutation["name"],
                "module": mutation["module"],
                "note": mutation["note"],
                "test": mutation["test"],
                "caught": caught,
                "detail": tail,
            }
        )
        print(f"  [{'CAUGHT' if caught else 'MISSED'}] {mutation['name']} -> {mutation['test']}")

    report = {
        "kind": "truncation mutation record; every named mutation must be caught",
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "suite": SUITE,
        "baseline_green": baseline_code == 0,
        "mutations": records,
        "caught": sum(1 for record in records if record["caught"]),
        "total": len(records),
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"mutation record: {out}")
    missed = [record for record in records if not record["caught"]]
    print(f"{report['caught']}/{report['total']} mutations caught")
    return 1 if missed else 0


if __name__ == "__main__":
    raise SystemExit(main())
