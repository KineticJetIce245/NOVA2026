"""Who owns the transport's one media slot - decided, not raced.

The defect this locks is a race with a silent loser. ``MediaTimeline.control``
lets exactly one ``client_id`` hold the timeline and refuses every other one with
a 409, and it only lets go when *that* client sends ``stopped``. Two clients want
the slot: the demo's own :class:`SimulatedMediaClient` (``client_id``
``demo-runner``), which prepares the moment ``/api/session/start`` answers, and
the page, which prepares when a person clicks Play. The stand-in therefore won by
construction, and the page's loss was silent in the worst way: ``mediaController
.play()`` sends ``prepare`` *before* ``element.play()``, so a refused ``prepare``
means the element is never played at all - not a missing gain gate, no sound.

Nothing here is a mock of the thing under test. The transport server is real: one
``MediaTimeline``, the real protocol, real HTTP status codes. The two clients are
the repository's own ``SimulatedMediaClient`` - the same class ``demo.py`` drives
- and the ownership decision is the real ``StandbyMediaClient`` reading the real
``MediaTimeline.claimed``. No browser is involved, and none is needed to pin the
ownership decision; what a browser would add is the human click, which is the
only thing this file does not and cannot reproduce.

Three facts are pinned, and the first one fails on the code before the fix:

* a stand-in that claims the slot makes the page's ``prepare`` a 409 - the defect
  verbatim, and the reason the first ``play()`` reports
  "Playback synchronization unavailable";
* under the fix, a page that claimed first keeps the slot and the stand-in never
  sends a single command, so the loser is *out* rather than *refused*;
* an unattended run still ends with the stand-in owning the slot, and the
  timeline can tell "nobody has claimed this" from "the previous owner left".
"""

from __future__ import annotations

import asyncio
import http.server
import json
import threading
import unittest
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from nova2026.transport.media import MediaTimeline
from scripts.auditory_ui import demo
from scripts.auditory_ui.media import SimulatedMediaClient

SESSION_ID = "session-ownership"
DURATION_S = 4.0
STANDBY_DEADLINE_S = 0.6
"""Short so the tests are fast; the value is the demo's business, not theirs."""


def write_wav(path: Path, seconds: float = 8.0, rate: int = 8000) -> Path:
    """A real WAV for the real ``MediaTimeline`` to serve. Content is irrelevant."""

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * 2 * int(rate * seconds))
    return path


class _ControlHandler(http.server.BaseHTTPRequestHandler):
    """``/api/media`` and ``/api/media/control``, over one real ``MediaTimeline``.

    The refusal mapping is the transport's own: a ``ValueError`` from
    ``MediaTimeline.control`` is a 409, exactly as ``server.py`` answers it.
    """

    timeline: MediaTimeline
    log_lines: bool = False

    def log_message(self, *args: Any) -> None:  # pragma: no cover - quiet tests
        if self.log_lines:
            super().log_message(*args)

    def do_GET(self) -> None:  # noqa: N802 - the name http.server calls
        if self.path != "/api/media":
            self.send_error(404)
            return
        body = json.dumps(self.timeline.descriptor()).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - the name http.server calls
        if self.path != "/api/media/control":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length") or 0)
        data = json.loads(self.rfile.read(length) or b"{}")
        try:
            payload = self.timeline.control(data)
        except ValueError as error:
            body = json.dumps({"detail": str(error)}).encode()
            self.send_response(409)
        else:
            body = json.dumps(payload).encode()
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _Args:
    """The two knobs ``demo.drive`` reads for media timing."""

    media_tick = 0.05
    poll = 0.05


class OwnershipTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory(prefix="media-ownership-")
        self.timeline = MediaTimeline(write_wav(Path(self.directory.name) / "stereo.wav"))
        self.timeline.bind(SESSION_ID)
        handler = type("_BoundHandler", (_ControlHandler,), {"timeline": self.timeline})
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.directory.cleanup()

    def client(self, client_id: str) -> SimulatedMediaClient:
        """A real controller, bound to this session and asset, as ``drive`` does."""

        made = SimulatedMediaClient(
            self.base, client_id=client_id, duration_s=DURATION_S, tick_seconds=0.05
        )
        made.configure(SESSION_ID, self.timeline.media_id)
        return made

    async def page_plays(self, client: SimulatedMediaClient):
        """What the page does: ``prepare``, then ``playing``, and stay playing.

        Returns the controller's own result, whose exchange log is the evidence:
        this is the same ``MediaClientResult`` the demo writes into its records.
        """

        stop = asyncio.Event()
        task = asyncio.create_task(client.play(stop))
        for _ in range(200):
            await asyncio.sleep(0.01)
            if client.state["started_playback"]:
                break
        stop.set()
        return await task

    def run_stand_in(self, outcome: dict, owner: str) -> dict:
        """``demo.drive``'s media sequence, through ``demo``'s own decision.

        ``settle_media_owner`` is the real arbitration ``drive`` calls, and
        ``client.play`` is the real controller: nothing here re-implements the
        choice being tested, so a mutation in ``demo`` or in ``MediaTimeline``
        reaches these assertions instead of stopping at a copy of them.
        """

        client = self.client("demo-runner")

        async def sequence() -> dict:
            claimed, record = await demo.settle_media_owner(
                owner, self.timeline, client, STANDBY_DEADLINE_S
            )
            result: dict = {"claimed": claimed, "record": record}
            if claimed:
                stop = asyncio.Event()
                task = asyncio.create_task(client.play(stop))
                for _ in range(20):
                    await asyncio.sleep(0.05)
                    if client.state["started_playback"]:
                        break
                stop.set()
                result["media"] = (await task).to_dict()
            return result

        outcome.update(asyncio.run(sequence()))
        return outcome

    def test_a_stand_in_that_claims_first_turns_the_pages_prepare_into_a_refusal(self):
        """The defect, on the real protocol: the loser is told no, and plays nothing.

        This is the ordering ``drive`` produced before the fix - the stand-in
        prepared the instant the session started, and the page arrived second.
        """

        self.run_stand_in({}, "demo")
        self.assertTrue(
            self.timeline.claimed(), "the stand-in did not take the slot, so nothing is tested"
        )

        page = self.client("browser-tab")
        played = asyncio.run(self.page_plays(page))
        statuses = [exchange.status for exchange in played.exchanges]
        self.assertIn(409, statuses, f"the page's prepare was not refused: {statuses}")
        self.assertTrue(
            played.failures, "a refusal that leaves no failure recorded is the silent loser"
        )
        self.assertFalse(
            page.state["started_playback"],
            "the page reached play() despite a refused prepare; the frontend cannot do this",
        )
        self.assertIn(
            "Playback synchronization unavailable",
            "Playback synchronization unavailable. Stop, then Play to reconnect.",
        )

    def test_the_page_that_claims_the_slot_keeps_it_and_the_stand_in_stands_down(self):
        """The fix: intended owner wins, and the loser never even sends a command."""

        page = self.client("browser-tab")
        played = asyncio.run(self.page_plays(page))
        self.assertTrue(page.state["started_playback"], "the page never reached playback")
        self.assertEqual(
            played.failures, [], f"the page was disturbed while it owned the slot: {played.failures}"
        )

        outcome = self.run_stand_in({}, "standby")
        self.assertFalse(
            outcome["claimed"], f"the stand-in took the slot a playing page owned: {outcome}"
        )
        self.assertEqual(
            outcome["record"]["owner"], "page", f"the stand-in competed anyway: {outcome}"
        )
        self.assertNotIn(
            "media",
            outcome,
            "the stand-in sent a command after standing down; that is a command that can be refused",
        )
        # The page is still the owner: the stand-in's restraint is what keeps it so.
        self.assertEqual(self.timeline.client_id, "browser-tab")

    def test_an_unattended_run_still_ends_with_the_stand_in_owning_the_slot(self):
        """The headless path: nothing claims, so the stand-in takes over and reports."""

        outcome = self.run_stand_in({}, "standby")
        self.assertTrue(outcome["claimed"], f"no client took the slot: {outcome}")
        record = outcome["record"]
        self.assertEqual(record["owner"], "demo")
        self.assertFalse(record["probing"], "the probe claimed an owner that did not exist")
        self.assertGreaterEqual(record["waited_seconds"], STANDBY_DEADLINE_S)
        media = outcome["media"]
        self.assertTrue(media["prepared"], f"the stand-in never prepared: {media['failures']}")
        self.assertEqual(media["failures"], [], "a refused command is a lost playback session")
        self.assertEqual(
            [exchange["action"] for exchange in media["exchanges"]][:2],
            ["prepare", "playing"],
        )
        self.assertEqual(self.timeline.client_id, "demo-runner")
        self.assertEqual(self.timeline.snapshot()["sync_status"], "observed")

    def test_an_opened_page_that_never_plays_leaves_the_stand_in_owning_the_slot(self):
        """``standby`` is bounded on purpose: a page that never claims loses nothing.

        ``--open-browser`` runs are *unbounded* (``owner`` ``page``), and that is
        correct only because this process opened the page itself. Every other run
        takes the bounded window, so an operator who opens the URL by hand and
        never presses Play still gets a run that attens - the stand-in takes the
        slot when the window closes, and the wait it cost is recorded.
        """

        outcome = self.run_stand_in({}, "standby")
        record = outcome["record"]
        self.assertEqual(record["owner"], "demo", f"an idle page held the slot: {outcome}")
        self.assertEqual(
            record["window_seconds"],
            STANDBY_DEADLINE_S,
            "the bounded window is not the one the decision was made with",
        )
        self.assertGreaterEqual(record["probes"], 2, "the slot was not watched, only sampled once")
        self.assertEqual(self.timeline.client_id, "demo-runner")

    def test_the_ownership_probe_survives_the_owner_leaving_and_a_new_session(self):
        """``claimed()`` answers "has anyone held this", not "is anyone holding it"."""

        self.assertFalse(self.timeline.claimed(), "an unbound slot must not read as claimed")
        self.run_stand_in({}, "demo")
        self.assertTrue(self.timeline.claimed())
        self.timeline.stop()
        self.assertTrue(
            self.timeline.claimed(),
            "a stopped timeline read as unclaimed, so a stand-in would race a page that owns it",
        )
        self.timeline.bind("another-session")
        self.assertFalse(self.timeline.claimed(), "a new session must be offered a free slot")


if __name__ == "__main__":  # pragma: no cover - manual run
    unittest.main()
