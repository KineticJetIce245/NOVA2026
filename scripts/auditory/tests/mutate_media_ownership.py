"""Apply, or undo, the mutations each ownership test is meant to catch.

Usage::

    python -B scripts/auditory/tests/mutate_media_ownership.py apply [race|probe]
    python -B scripts/auditory/tests/mutate_media_ownership.py revert

Two mutations, and neither can pass silently:

``race`` (default)
    ``demo.settle_media_owner`` stops arbitrating and reports "the stand-in owns
    it" - which is what the code did before the fix, and what made the page lose
    the race. Expected to turn ``test_the_page_that_claims_the_slot_keeps_it_*``
    and ``test_an_opened_page_that_never_plays_*`` red.

``probe``
    ``MediaTimeline.claimed`` stops being sticky: a slot whose owner has gone
    reads as never claimed, which is exactly the confusion that lets a stand-in
    race a page that owns the slot. Expected to turn
    ``test_the_page_that_claims_the_slot_keeps_it_*`` and
    ``test_the_ownership_probe_survives_*`` red.

Both are applied in place and reverted byte-for-byte; the file's own hash is
printed on the way back so a half-applied mutation cannot be left behind.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEMO = REPO / "scripts" / "auditory_ui" / "demo.py"
TIMELINE = REPO / "src" / "nova2026" / "transport" / "media.py"

MUTATIONS: dict[str, tuple[Path, str, str]] = {
    "race": (
        DEMO,
        "    if owner == \"demo\":\n        return True, None\n",
        "    # MUTATION: the arbitration is gone; the stand-in always claims the slot.\n"
        "    return True, None\n",
    ),
    "probe": (
        TIMELINE,
        "        with self.lock:\n            return self.ever_claimed_at is not None\n",
        "        # MUTATION: not sticky, so a departed owner reads as a free slot.\n"
        "        with self.lock:\n            return self.client_id is not None\n",
    ),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str]) -> int:
    action = argv[1] if len(argv) > 1 else ""
    name = argv[2] if len(argv) > 2 else "race"
    if name not in MUTATIONS:
        print(f"unknown mutation {name!r}; expected one of {sorted(MUTATIONS)}", file=sys.stderr)
        return 2
    path, before, after = MUTATIONS[name]
    original = path.read_text(encoding="utf-8")
    if action == "apply":
        if after in original:
            print(f"already mutated: {name}")
            return 0
        if before not in original:
            print(f"the code {name!r} mutates is not in {path.name}; stale mutation", file=sys.stderr)
            return 2
        path.write_text(original.replace(before, after, 1), encoding="utf-8", newline="")
        print(f"mutated {path.name}: {name}")
        return 0
    if action == "revert":
        if after not in original:
            print(f"not mutated: {name}")
            return 0
        path.write_text(original.replace(after, before, 1), encoding="utf-8", newline="")
        print(f"reverted {path.name}: sha256 {digest(path)}")
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

