"""The test runner's own rules, which are what stop a suite going quiet.

``tests/visual_detect`` held 56 tests that ``unittest discover`` reported as
"Ran 0 tests ... OK" for months, because they are plain functions rather than
``TestCase`` classes. The runner exists to make that impossible, so the rule that
does it - an empty suite is a failure, not a pass - is worth a test of its own.
"""

import unittest
from pathlib import Path

from scripts.run_tests import Result, Suite, commands, parse

UNITTEST_OK = """
....................
----------------------------------------------------------------------
Ran 42 tests in 0.031s

OK
"""

UNITTEST_FAILED = """
======================================================================
FAIL: test_something (tests.x.Thing)
----------------------------------------------------------------------
Ran 42 tests in 0.031s

FAILED (failures=1)
"""

FUNCTIONS_OK = "PASS test_a\nPASS test_b\n\n2 passed, 0 failed\n"

FUNCTIONS_FAILED = "PASS test_a\nFAIL test_b\n\n1 passed, 1 failed\n"

ALL_SKIPPED = """
....................
----------------------------------------------------------------------
Ran 5 tests in 0.010s

OK (skipped=5)
"""

SILENT_ZERO = "no tests ran here at all\n"


class ParseTests(unittest.TestCase):
    """Counting has to be exact: the summary is read as evidence."""

    def test_unittest_output_is_read(self) -> None:
        suite = Suite("streaming", "tests/streaming")
        self.assertEqual(parse(suite, UNITTEST_OK), (42, 0, 0))
        self.assertEqual(
            parse(suite, UNITTEST_FAILED),
            (42, 1, 0),
            "one failure is one failure, not a whole suite of them",
        )

    def test_errors_and_failures_both_count(self) -> None:
        suite = Suite("streaming", "tests/streaming")
        mixed = "Ran 9 tests in 0.1s\n\nFAILED (failures=2, errors=3, skipped=1)\n"
        self.assertEqual(parse(suite, mixed), (9, 5, 1))

    def test_skips_are_read_separately_from_failures(self) -> None:
        suite = Suite("streaming", "tests/streaming")
        self.assertEqual(parse(suite, ALL_SKIPPED), (5, 0, 5))

    def test_function_suite_output_is_read(self) -> None:
        suite = Suite("visual-detect", "tests/visual_detect", functions=True)
        self.assertEqual(parse(suite, FUNCTIONS_OK), (2, 0, 0))
        self.assertEqual(parse(suite, FUNCTIONS_FAILED), (2, 1, 0))

    def test_several_function_modules_are_summed(self) -> None:
        suite = Suite("visual-detect", "tests/visual_detect", functions=True)
        self.assertEqual(parse(suite, FUNCTIONS_OK + FUNCTIONS_FAILED), (4, 1, 0))

    def test_output_without_a_count_is_not_a_pass(self) -> None:
        suite = Suite("streaming", "tests/streaming")
        self.assertEqual(parse(suite, SILENT_ZERO), (0, 0, 0))


class VerdictTests(unittest.TestCase):
    """The rule the runner exists for."""

    def test_an_empty_suite_is_a_failure(self) -> None:
        result = Result(Suite("streaming", "tests/streaming"), 0, 0, 0.1, "")
        self.assertFalse(result.ok)
        self.assertEqual(result.problem, "reported no tests at all")

    def test_a_passing_suite_is_ok(self) -> None:
        result = Result(Suite("streaming", "tests/streaming"), 42, 0, 0.1, "")
        self.assertTrue(result.ok)
        self.assertEqual(result.problem, "")

    def test_a_fully_skipped_suite_is_a_failure(self) -> None:
        result = Result(Suite("streaming", "tests/streaming"), 5, 0, 0.1, "", 5)
        self.assertFalse(result.ok)
        self.assertEqual(result.problem, "every one of 5 tests was skipped")

    def test_a_partly_skipped_suite_is_still_ok(self) -> None:
        result = Result(Suite("streaming", "tests/streaming"), 5, 0, 0.1, "", 1)
        self.assertTrue(result.ok)

    def test_failures_and_errors_are_reported(self) -> None:
        suite = Suite("streaming", "tests/streaming")
        self.assertFalse(Result(suite, 42, 1, 0.1, "").ok)
        self.assertEqual(Result(suite, 42, 1, 0.1, "").problem, "1 failed")
        crashed = Result(suite, 0, 0, 0.1, "", error="exited non-zero: boom")
        self.assertFalse(crashed.ok)
        self.assertIn("boom", crashed.problem)


class CommandTests(unittest.TestCase):
    def test_a_unittest_suite_is_discovered(self) -> None:
        command = commands(Suite("streaming", "tests/streaming"))
        self.assertEqual(len(command), 1)
        self.assertIn("unittest", command[0])
        self.assertIn("discover", command[0])

    def test_a_function_suite_runs_each_module(self) -> None:
        found = commands(Suite("visual-detect", "tests/visual_detect", functions=True))
        self.assertEqual(len(found), 3, "one process per test module")
        for command in found:
            self.assertTrue(command[-1].endswith(".py"), command)
            self.assertIn("test_", Path(command[-1]).name)


if __name__ == "__main__":
    unittest.main()
