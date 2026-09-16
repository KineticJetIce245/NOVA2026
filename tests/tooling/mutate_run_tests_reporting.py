"""Show that the reporting tests fail under a named mutation, then restore.

Plan section 6.4 requires it, and this file's subject is a case in point: a test
of a *report* passes trivially if it asserts something the old report also
printed. Each mutation below puts the old behaviour back - the ten-line tail, and
the tail-printer in the summary - and the test that is supposed to pin the new
behaviour must go red.

    .venv\\Scripts\\python.exe -B tests\\tooling\\mutate_run_tests_reporting.py

The mutated module is ``scripts/run_tests.py``: it lives under ``scripts/`` and is
imported from the working directory, not through the editable install that points
at ``src/``, so a subprocess started here really does import the mutated line -
a mutation that was not seen would be reported as MISSED. The file is restored in
a ``finally`` block and its bytes are compared before and after, so a run that
dies half way cannot leave a mutated tree behind.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SUITE = "tests/tooling"
MODULE = "scripts/run_tests.py"

MUTATIONS = (
    {
        "name": "print the last ten lines again instead of the failure headers",
        "find": (
            "    lines = output.splitlines()\n"
            "    headers = [index for index, line in enumerate(lines)"
            " if UNITTEST_FAILURE.match(line)]\n"
            "    if not headers:\n"
            "        return lines[-tail:]\n"
            "    picked: list[str] = []\n"
            "    for index in headers:\n"
            "        picked.extend(lines[index : index + context + 1])\n"
            "    return picked"
        ),
        "replace": "    return output.splitlines()[-tail:]",
        "test": "tests.tooling.test_run_tests.FailureReportTests.test_every_failure_header_is_named",
        "note": "the report as it was: eight minutes of output, ten lines of it printed",
    },
    {
        "name": "drop the header report from the summary",
        "find": (
            "            named = UNITTEST_FAILURE.search(result.output) is not None\n"
            '            print("    failing tests:" if named else "    last lines:")\n'
            "            for line in failure_report(result.output):\n"
            '                print(f"      {line}")'
        ),
        "replace": (
            '            print("    last lines:")\n'
            "            for line in result.output.strip().splitlines()[-10:]:\n"
            '                print(f"      {line}")'
        ),
        "test": "tests.tooling.test_run_tests.ReportedFailureTests.test_a_red_suite_prints_the_failing_test",
        "note": "the helper is fine, but nothing calls it: the summary keeps its tail",
    },
)


def digest(path: Path) -> str:
    """SHA256 of one file, so a mutation cannot be left behind unnoticed."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_suite() -> tuple[int, str, list[str]]:
    """Run the tooling suite and return its exit code, a tail, and its red names."""

    completed = subprocess.run(
        [sys.executable, "-B", "-m", "unittest", "discover", "-s", SUITE],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    text = completed.stdout + completed.stderr
    red = [
        line.strip()
        for line in text.splitlines()
        if line.startswith(("FAIL:", "ERROR:"))
    ]
    tail = [line for line in text.splitlines() if line.strip()]
    return completed.returncode, " | ".join(tail[-3:])[:400], red


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/run_tests_reporting_mutation.json")
    args = parser.parse_args(argv)

    baseline_code, baseline_tail, _ = run_suite()
    records = []
    if baseline_code != 0:
        print(f"the unmutated suite is already red; nothing to prove: {baseline_tail}")
        return 1
    print(f"baseline: {SUITE} is green")

    for mutation in MUTATIONS:
        path = REPO / MODULE
        before = digest(path)
        # ``newline=""`` on both sides: a text-mode round trip would translate
        # every LF into CRLF, which is a whole-file diff and a false alarm.
        source = path.read_text(encoding="utf-8", newline="")
        # This file is stored with CRLF - the other mutation target in this
        # repository, ``scripts/auditory_ui/demo.py``, is LF - so a pattern joined
        # with "\n" silently finds nothing. The first run of this script reported
        # both mutations as "the code moved" for exactly that reason. The find and
        # replace texts are translated to whatever this file actually uses, and the
        # file is written back with ``newline=""`` so nothing is normalised.
        ending = "\r\n" if "\r\n" in source else "\n"
        find = mutation["find"].replace("\n", ending)
        replace = mutation["replace"].replace("\n", ending)
        if find not in source:
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
            source.replace(find, replace, 1), encoding="utf-8", newline=""
        )
        try:
            code, tail, red = run_suite()
        finally:
            path.write_text(source, encoding="utf-8", newline="")
            restored = digest(path) == before
        if not restored:
            print(f"FATAL: {MODULE} was not restored")
            return 2
        # Caught means: the suite went red *and* the red run named the test this
        # mutation is supposed to kill. A mutation that merely breaks the import
        # would otherwise be recorded as a caught mutation.
        name = mutation["test"].rsplit(".", 1)[-1]
        caught = code != 0 and any(name in line for line in red)
        records.append(
            {
                "mutation": mutation["name"],
                "module": MODULE,
                "note": mutation["note"],
                "test": mutation["test"],
                "caught": caught,
                "red": red,
                "detail": tail,
            }
        )
        print(f"  [{'CAUGHT' if caught else 'MISSED'}] {mutation['name']}")
        for line in red[:4]:
            print(f"      red: {line}")

    report = {
        "kind": "run_tests reporting mutation record; every named mutation must be caught",
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
