"""Show that each step-9 test fails under a named mutation, then restore the tree.

Plan section 6.4 requires it: a test that passes is not yet evidence that it
tests anything, so every mutation listed below is applied to the *library* code,
the suite is run, and the suite must go red. The mutation is reverted in a
``finally`` block, and the file's hash is compared before and after, so a run that
dies half way cannot leave a broken tree behind silently.

    .venv\\Scripts\\python.exe -B scripts\\auditory\\mutate_media.py [--out results/...json]

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

REPO = Path(__file__).resolve().parents[2]
SUITE = "scripts.auditory.tests.test_media_timeline"

MUTATIONS = (
    {
        "name": "swap the two output channels",
        "module": "src/nova2026/auditory/render.py",
        "find": "    if mode == \"dichotic\":\n        return np.clip(samples, -1.0, 1.0)",
        "replace": "    if mode == \"dichotic\":\n        return np.clip(samples[:, ::-1], -1.0, 1.0)",
        "test": "RenderTests.test_dichotic_puts_candidate_a_on_the_left_and_b_on_the_right",
    },
    {
        "name": "scale to the array's peak instead of the file's own level",
        "module": "src/nova2026/auditory/render.py",
        "find": "    scaled = values * INT16_PEAK",
        "replace": (
            "    peak = float(np.max(np.abs(values))) if values.size else 1.0\n"
            "    scaled = values / peak * INT16_PEAK"
        ),
        "test": "RenderTests.test_dichotic_puts_candidate_a_on_the_left_and_b_on_the_right",
    },
    {
        "name": "accept an unknown presentation mode as the default",
        "module": "src/nova2026/auditory/render.py",
        "find": "    if name not in PRESENTATION_MODES:",
        "replace": "    if False:",
        "test": "RenderTests.test_the_two_modes_are_selectable_and_an_unknown_mode_is_refused",
    },
    {
        "name": "return a zeroed reference before any controller prepared",
        "module": "src/nova2026/transport/media.py",
        "find": (
            "            if not self.valid or self.last_received is None:\n"
            "                return None"
        ),
        "replace": "",
        "test": "ReferenceTests.test_no_reference_before_a_controller_prepares_the_timeline",
    },
    {
        "name": "drop the freshness test, so a dead tab stays synchronized",
        "module": "src/nova2026/transport/media.py",
        "find": "            if self.clock() - self.last_received > FRESH_SECONDS:\n                return None",
        "replace": "",
        "test": "ReferenceTests.test_a_stale_report_withdraws_the_reference",
    },
    {
        "name": "repair an invalidated timeline on the next accepted command",
        "module": "src/nova2026/transport/media.py",
        "find": (
            "                    self.valid = False\n"
            "                    raise ValueError("
        ),
        "replace": (
            "                    self.valid = False\n"
            "                    self.valid = True  # mutant\n"
            "                    raise ValueError("
        ),
        "test": "TimelineRefusalTests.test_a_jump_beyond_the_elapsed_time_is_refused_and_is_sticky",
    },
    {
        "name": "allow a full second of jump while playing",
        "module": "src/nova2026/transport/media.py",
        "find": "PLAYING_ADVANCE_ALLOWANCE = 0.5",
        "replace": "PLAYING_ADVANCE_ALLOWANCE = 1.0",
        "test": "TimelineRefusalTests.test_just_over_half_a_second_ahead_is_refused_while_playing",
    },
    {
        "name": "keep the revision unchanged when playback stops",
        "module": "src/nova2026/transport/media.py",
        "find": "            self.valid = False\n            self.revision += 1",
        "replace": "            self.valid = False",
        "test": "TimelineRefusalTests.test_stopping_is_terminal_for_that_revision",
    },
    {
        "name": "read the timeline twice per frame",
        "module": "src/nova2026/auditory/producer.py",
        "find": (
            "        media = self._media()\n"
            "        attention = self._attention(frame)\n"
            "        attention.update(media)\n"
            "        gain = self._gain(frame)\n"
            "        gain.update(media)"
        ),
        "replace": (
            "        attention = self._attention(frame)\n"
            "        attention.update(self._media())\n"
            "        gain = self._gain(frame)\n"
            "        gain.update(self._media())"
        ),
        "test": "ProducerEchoTests.test_one_frame_reads_the_timeline_once_and_stamps_both_packets",
    },
    {
        "name": "publish a frame's requested gain without the attenuation ceiling",
        "module": "src/nova2026/auditory/producer.py",
        "find": "            \"a_db\": min(0.0, frame.gain_a_db),",
        "replace": "            \"a_db\": frame.gain_a_db,",
        "test": "ProducerEchoTests.test_gain_never_exceeds_zero_db_even_if_a_frame_asks_for_amplification",
    },
    {
        "name": "publish the media packet regardless of the session's status",
        "module": "src/nova2026/transport/media.py",
        "find": (
            "        session_id = (\n"
            "            records[-1][\"id\"]\n"
            "            if records and records[-1][\"status\"] == \"running\"\n"
            "            else None\n"
            "        )"
        ),
        "replace": "        session_id = records[-1][\"id\"] if records else None",
        "test": "BroadcasterTests.test_no_packet_is_published_without_a_running_session",
    },
    {
        "name": "publish the raw clock as the media packet's timestamp",
        "module": "src/nova2026/transport/media.py",
        "find": "        return max(0.0, now - self._baseline)",
        "replace": "        return now",
        "test": "BroadcasterTests.test_timestamps_do_not_regress_within_the_stream",
    },
)


def digest(path: Path) -> str:
    """SHA256 of one file, so a mutation cannot be left behind unnoticed."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_suite(test: str) -> tuple[int, str]:
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
    parser.add_argument("--out", default="results/media_mutations.json")
    args = parser.parse_args(argv)

    baseline_code, baseline_tail = run_suite(SUITE)
    records = []
    if baseline_code != 0:
        print(f"the unmutated suite is already red; nothing to prove: {baseline_tail}")
        return 1
    print(f"baseline: {SUITE} is green")

    for mutation in MUTATIONS:
        path = REPO / mutation["module"]
        # ``newline=""`` on both sides: a text-mode round trip would translate
        # every LF into CRLF, which is a whole-file diff and a false alarm.
        original = path.read_bytes()
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
        path.write_text(source.replace(mutation["find"], mutation["replace"], 1), encoding="utf-8", newline="")
        try:
            code, tail = run_suite(mutation["test"])
        finally:
            path.write_text(source, encoding="utf-8", newline="")
            # Read the bytes back rather than hashing the text: on Windows a
            # text-mode round trip can normalise line endings, and a false
            # "not restored" would be as bad as a real one.
            restored = path.read_bytes() == original
        if not restored:
            print(f"FATAL: {mutation['module']} was not restored")
            return 2
        caught = code != 0
        records.append(
            {
                "mutation": mutation["name"],
                "module": mutation["module"],
                "test": mutation["test"],
                "caught": caught,
                "detail": tail,
            }
        )
        print(f"  [{'CAUGHT' if caught else 'MISSED'}] {mutation['name']} -> {mutation['test']}")

    report = {
        "kind": "step-9 mutation record; every named mutation must be caught",
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
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"mutation record: {out}")
    missed = [record for record in records if not record["caught"]]
    print(f"{report['caught']}/{report['total']} mutations caught")
    return 1 if missed else 0


if __name__ == "__main__":
    raise SystemExit(main())
