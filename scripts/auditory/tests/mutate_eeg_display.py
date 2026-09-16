"""Apply one named mutation to the display tap, run the tests, restore the file.

Usage: python -B scripts/auditory/tests/mutate_eeg_display.py <mutation>

The file is restored from an in-memory copy in a finally block, and the restore is
verified by comparing the file's bytes, so a crash mid-mutation cannot leave the
tree altered. Each mutation is one that a plausible wrong implementation would
have, and the test named beside it is the one that must go red.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TARGET = REPO / "src" / "nova2026" / "auditory" / "streaming.py"
TESTS = REPO / "tests" / "streaming" / "test_eeg_display.py"

MUTATIONS = {
    # Hands out a view of the ring's own storage instead of a copy. This is the
    # aliasing the capture exists to prevent: the next chunk overwrites exactly
    # the cells a captured window is holding.
    "no_copy": (
        "            return self._ring[offset : offset + window_samples].copy()",
        "            return self._ring[offset : offset + window_samples]",
    ),
    # Adds a key to the frozen contract (invalidates every trained decoder).
    "contract_key": (
        '"window_seconds": settings.window_seconds, "step_seconds": settings.step_seconds,',
        '"window_seconds": settings.window_seconds, "step_seconds": settings.step_seconds,\n'
        '            "display_channel": self._display_channel,',
    ),
    # A bare slice: aliases, and its length varies with the window length.
    "bare_slice": (
        "    grouped = array.reshape(array.shape[0] // factor, factor, *array.shape[1:])",
        "    grouped = array[::factor]",
    ),
    # Relabels the pre-band-pass rate as the post-resample one.
    "swapped_rate": (
        "        sample_rate=float(source_rate),",
        "        sample_rate=float(filtered_rate),",
    ),
    # Publishes the display packet unconditionally.
    "always_publish": (
        "if display:",
        "if True:",
    ),
    # One factor for both traces, as the first implementation had: the point
    # counts then differ by the resampling ratio. Patched at the shared helper so
    # this mutation and the code it mutates cannot drift apart.
    "shared_factor": (
        "    ceiling = int(max(1.0, float(target_points)))",
        "    return raw_size // decimation_factor(raw_size, target_points), "
        "filtered_size // decimation_factor(raw_size, target_points), "
        "raw_size // decimation_factor(raw_size, target_points)\n"
        "    ceiling = int(max(1.0, float(target_points)))",
    ),
}


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in MUTATIONS:
        print("usage: mutate_eeg_display.py " + "|".join(sorted(MUTATIONS)))
        return 2
    name = argv[1]
    old, new = MUTATIONS[name]
    original = TARGET.read_bytes()
    text = original.decode("utf-8")
    if text.count(old) != 1:
        print(f"anchor for {name!r} appears {text.count(old)} times; refusing to guess.")
        return 2
    print(f"=== mutation {name}: replacing\n{old}\n--- with\n{new}")
    try:
        TARGET.write_text(text.replace(old, new), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "-B", "-m", "unittest", "tests.streaming.test_eeg_display"],
            cwd=REPO, capture_output=True, text=True,
        )
        tail = [line for line in result.stderr.splitlines() if line.strip()]
        print("\n".join(tail[-14:]))
        print(f"exit code {result.returncode} (0 would mean the mutation survived)")
    finally:
        TARGET.write_bytes(original)
        restored = TARGET.read_bytes() == original
        print(f"restored: {restored}")
    return 0 if restored else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
