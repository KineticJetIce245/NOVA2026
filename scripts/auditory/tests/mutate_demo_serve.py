"""Show that the stay-open tests fail under a named mutation, then restore.

Plan section 6.4 requires it: a test that passes is not yet evidence that it
tests anything. Two behaviours are locked by ``test_demo_serve``, and each
mutation below breaks one of them the way the code actually was broken - the
first is the shipped defect verbatim (``stop.set()`` at the end of the drive
phase), the second is its opposite (a stay-open loop that ignores the stop).

    .venv\\Scripts\\python.exe -B scripts\\auditory\\tests\\mutate_demo_serve.py

Each mutation names the test it must kill. The file is restored in a ``finally``
block and its bytes are compared before and after, so a run that dies half way
cannot leave a mutated tree behind silently. The mutated module lives under
``scripts/`` - imported from the working directory, not through the editable
install - so a subprocess started here really does import the mutated line; the
run would report the mutation as MISSED if it did not.
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
SUITE = "scripts.auditory.tests.test_demo_serve"

MUTATIONS = (
    {
        "name": "signal the run's stop event when the replay ends",
        "module": "scripts/auditory_ui/demo.py",
        "find": '            finished.set()\n            state["media"] = (await playback).to_dict()',
        "replace": '            stop.set()\n            state["media"] = (await playback).to_dict()',
        "test": f"{SUITE}.ServePhaseTests"
        ".test_the_transport_keeps_answering_after_the_drive_phase_returns",
        "note": "the code as it stood: the stay-open loop exits on its first check",
    },
    {
        "name": "ignore the stop request in the stay-open loop",
        "module": "scripts/auditory_ui/demo.py",
        "find": "            while not stop.is_set():\n"
        "                if deadline is not None and time.monotonic() >= deadline:\n"
        "                    break",
        "replace": "            while True:\n"
        "                if deadline is not None and time.monotonic() >= deadline:\n"
        "                    break",
        "test": f"{SUITE}.ServePhaseTests.test_the_run_ends_when_the_operator_stops_it",
        "note": "a fix that keeps serving by never listening for the stop",
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
        timeout=600,
        check=False,
    )
    text = completed.stdout + completed.stderr
    tail = [line for line in text.splitlines() if line.strip()]
    # The tail is where the assertion or the traceback is; stderr is printed
    # rather than dropped, so a mutation that fails to import is not mistaken
    # for a caught mutation (plan D-44).
    return completed.returncode, " | ".join(tail[-4:])[:500]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/demo_serve_mutation.json")
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
            print(f"  [MISSED] {mutation['name']} -> the code moved")
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
        "kind": "stay-open mutation record; every named mutation must be caught",
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
