"""Run every test suite in this repository, and say plainly what happened.

The tests live in four places and come in two styles, which is how one of them
came to never run at all: ``tests/visual_detect`` holds plain ``test_*``
functions with their own runners rather than ``unittest.TestCase`` classes, so
``unittest discover`` reported "Ran 0 tests ... OK" for it - green, and empty.

This runs all four, from the repository root, and refuses to call an empty suite
a pass. A suite that reports no tests is reported as a failure, because that is
the only way the silence cannot come back.

    .venv/bin/python -B scripts/run_tests.py
    .venv/bin/python -B scripts/run_tests.py --suite streaming -v

Exit codes: 0 every suite ran and passed, 1 otherwise.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# ``unittest`` prints "Ran N tests" and a summary line; the function suites
# print "N passed, M failed". Both are read here, and a suite that matches
# neither is an error rather than a zero.
UNITTEST_RAN = re.compile(r"^Ran (\d+) tests?", re.MULTILINE)
UNITTEST_SUMMARY = re.compile(r"^(OK|FAILED)(?: \(([^)]*)\))?", re.MULTILINE)
SUMMARY_COUNT = re.compile(r"(\w+)=(\d+)")
FUNCTION_RAN = re.compile(r"(\d+) passed, (\d+) failed")
# One ``unittest`` failure or error header. The names of the failures are here,
# and nowhere else: the summary's tail is whatever test happened to run last.
UNITTEST_FAILURE = re.compile(r"^(?:FAIL|ERROR): .*$", re.MULTILINE)

FAILURE_CONTEXT = 8
"""Lines printed under each failure header: the frames and the message under it."""


@dataclass(frozen=True)
class Suite:
    """One place with tests in it.

    Args:
        name: Short name for the summary and for ``--suite``.
        path: Directory holding the tests, relative to the repository root.
        functions: ``True`` for plain ``test_*`` function modules that run
            themselves, ``False`` for ``unittest`` discovery.
    """

    name: str
    path: str
    functions: bool = False

    @property
    def directory(self) -> Path:
        """Absolute path to the suite."""

        return REPO / self.path


SUITES = (
    Suite("streaming", "tests/streaming"),
    Suite("tooling", "tests/tooling"),
    Suite("transport", "tests/transport"),
    Suite("visual-detect", "tests/visual_detect", functions=True),
    Suite("auditory", "scripts/auditory/tests"),
    Suite("dataproc-streaming", "scripts/dataproc/streaming/tests"),
)


@dataclass
class Result:
    """What one suite did."""

    suite: Suite
    ran: int
    failed: int
    seconds: float
    output: str
    skipped: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        """Whether tests actually ran in this suite and all of them passed.

        A suite where every test skipped is not a pass either: it is the same
        silence as a suite with no tests, wearing a green badge.
        """

        return (
            self.error is None
            and self.failed == 0
            and self.ran > 0
            and self.ran > self.skipped
        )

    @property
    def problem(self) -> str:
        """One line saying what went wrong, or an empty string."""

        if self.error is not None:
            return self.error
        if self.ran == 0:
            return "reported no tests at all"
        if self.ran == self.skipped:
            return f"every one of {self.ran} tests was skipped"
        if self.failed:
            return f"{self.failed} failed"
        return ""


def environment() -> dict:
    """The environment every suite runs in: importable repo, quiet matplotlib."""

    env = dict(os.environ)
    path = [str(REPO)]
    if env.get("PYTHONPATH"):
        path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(path)
    env.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="nova-mpl-"))
    return env


def commands(suite: Suite) -> list[list[str]]:
    """The subprocess commands that run one suite.

    Function suites are one process per module, so a failure in one still lets
    the others report, and each keeps the runner its author wrote.
    """

    base = [sys.executable, "-B"]
    if not suite.functions:
        return [base + ["-m", "unittest", "discover", "-s", str(suite.directory)]]
    modules = sorted(suite.directory.glob("test_*.py"))
    return [base + [str(module)] for module in modules]


def parse(suite: Suite, output: str) -> tuple[int, int, int]:
    """Read ``(ran, failed, skipped)`` out of a suite's output."""

    if not suite.functions:
        ran = UNITTEST_RAN.search(output)
        if ran is None:
            return 0, 0, 0
        failed = skipped = 0
        summary = UNITTEST_SUMMARY.search(output)
        if summary is not None:
            counts = dict(SUMMARY_COUNT.findall(summary.group(2) or ""))
            failed = int(counts.get("failures", 0)) + int(counts.get("errors", 0))
            skipped = int(counts.get("skipped", 0))
            if summary.group(1) == "FAILED" and not counts:
                failed = 1
        return int(ran.group(1)), failed, skipped
    total = failed = 0
    for match in FUNCTION_RAN.finditer(output):
        total += int(match.group(1)) + int(match.group(2))
        failed += int(match.group(2))
    return total, failed, 0


def failure_report(output: str, context: int = FAILURE_CONTEXT, tail: int = 10) -> list[str]:
    """The lines that name a failing suite's failures, for the summary.

    ``unittest`` writes one ``FAIL:``/``ERROR:`` header per problem with the
    traceback under it. The last ten lines of a suite that ran for eight minutes
    are the tail of whichever test happened to run last, so they name nothing -
    and naming the failure is the one thing a red run has to do. Every header is
    printed instead, with the first frames and the message beneath it. Output with
    no header at all - an import error, a collection error, a timeout - keeps the
    tail, which is where those put their one useful line. Only what is *printed*
    changes: the commands, the parsing and the exit code are untouched.
    """

    lines = output.splitlines()
    headers = [index for index, line in enumerate(lines) if UNITTEST_FAILURE.match(line)]
    if not headers:
        return lines[-tail:]
    picked: list[str] = []
    for index in headers:
        picked.extend(lines[index : index + context + 1])
    return picked


def run(suite: Suite, verbose: bool, timeout: float) -> Result:
    """Run one suite and collect what it reported."""

    started = time.monotonic()
    collected: list[str] = []
    ran = failed = skipped = 0
    error: str | None = None
    for command in commands(suite):
        try:
            completed = subprocess.run(
                command,
                cwd=REPO,
                env=environment(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            error = f"timed out after {timeout:g}s"
            break
        collected.append(completed.stdout)
        if verbose:
            sys.stdout.write(completed.stdout)
            sys.stdout.flush()
        part_ran, part_failed, part_skipped = parse(suite, completed.stdout)
        ran += part_ran
        failed += part_failed
        skipped += part_skipped
        if completed.returncode != 0 and part_failed == 0:
            # Non-zero without a parsed failure: a crash, an import error, a
            # collection error. Report the tail, which is where it will be.
            error = "exited non-zero:\n" + "\n".join(
                completed.stdout.strip().splitlines()[-15:]
            )
            break
    return Result(
        suite, ran, failed, time.monotonic() - started, "".join(collected), skipped, error
    )


def main(argv: list[str] | None = None) -> int:
    """Run the suites and print the summary."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        action="append",
        choices=[suite.name for suite in SUITES],
        help="run one suite (repeatable); default is all of them",
    )
    parser.add_argument("--timeout", type=float, default=900.0, help="seconds per suite")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="stream each suite's output"
    )
    args = parser.parse_args(argv)

    chosen = [
        suite for suite in SUITES if not args.suite or suite.name in args.suite
    ]
    results = [run(suite, args.verbose, args.timeout) for suite in chosen]

    width = max(len(result.suite.name) for result in results)
    print(f"\n{'suite':<{width}}  {'tests':>6}  {'skip':>5}  {'result':<10}  time")
    for result in results:
        verdict = "ok" if result.ok else f"FAILED ({result.problem})"
        print(
            f"{result.suite.name:<{width}}  {result.ran:>6}  {result.skipped:>5}  "
            f"{verdict:<10}  {result.seconds:.1f}s"
        )
        if not result.ok and not args.verbose and result.output:
            # Names, not a tail: ten lines of an eight-minute suite are the last
            # test's output, and a red run that cannot say what failed costs
            # another eight minutes to ask again.
            named = UNITTEST_FAILURE.search(result.output) is not None
            print("    failing tests:" if named else "    last lines:")
            for line in failure_report(result.output):
                print(f"      {line}")

    total = sum(result.ran for result in results)
    skipped = sum(result.skipped for result in results)
    seconds = sum(result.seconds for result in results)
    bad = [result for result in results if not result.ok]
    print(
        f"\ntotal: {total} tests in {len(results)} suite(s), "
        f"{skipped} skipped, {seconds:.1f}s"
    )
    if bad:
        print("not green: " + ", ".join(result.suite.name for result in bad))
        return 1
    print("all green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
