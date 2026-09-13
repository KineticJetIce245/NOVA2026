"""The page outlives the replay: ``--serve-seconds`` and ``--browser`` really serve.

The defect this locks lived in one line of ``demo.drive``: its ``finally`` block
set the *same* event the stay-open loop waits on, so the loop exited on its first
check and the transport shut down in the second the replay ended - while the log
still said "still serving ... for 120s". Two behaviours are asserted, and they
fail in opposite directions, which is why a single assertion would not be enough:

* the replay reaching its terminal status must not stop the server - once
  ``drive`` returns, the port keeps answering ``/`` and ``/api/state`` for the
  whole ``--serve-seconds`` window;
* a real stop request must still end the run promptly - with ``--browser``
  (unbounded) the run returns as soon as the event Ctrl-C sets is signalled, so a
  fix that simply never stopped would fail here.

Plan section 6.4 also requires the test to be falsifiable:
``mutate_demo_serve.py`` in this directory restores each broken behaviour in turn
and shows which test goes red.

Nothing here needs a dataset, a model file or a browser: the fixture is generated
in a temporary directory, exactly as ``test_demo_entry``'s tests do.
"""

from __future__ import annotations

import asyncio
import http.client
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path

from scripts.auditory_ui import demo

REPO = Path(__file__).resolve().parents[3]
SERVE_SECONDS = 4.0
"""The bounded window the first test proves is served, second by second."""


def arguments(directory: Path, extra: list[str]) -> list[str]:
    """One synthetic run, with everything it writes kept inside ``directory``."""

    return [
        "--synthetic",
        "--seconds",
        "12",
        "--speed",
        "12",
        "--fixture-dir",
        str(directory / "fixture"),
        "--media-out",
        str(directory / "stereo.wav"),
        "--static-dir",
        str(REPO / "apps" / "attune-ui" / "dist"),
        "--stream-out",
        str(directory / "packets.jsonl"),
        "--render-out",
        str(directory / "dashboard.html"),
        "--media-record",
        str(directory / "media.json"),
        "--gate-out",
        str(directory / "gate.json"),
        "--out",
        str(directory / "demo_run.json"),
    ] + extra


def fetch(port: int) -> dict:
    """One browser-shaped probe of the running transport: the page, then the state.

    Both are asked for on one connection, in the order a person's tab would ask:
    the HTML first, then the state endpoint the dashboard polls.
    """

    record: dict = {"at": time.monotonic()}
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    try:
        connection.request("GET", "/")
        page = connection.getresponse()
        record["page"] = page.status
        record["page_bytes"] = len(page.read())
        connection.request("GET", "/api/state")
        state = connection.getresponse()
        record["state"] = state.status
        record["state_body"] = state.read().decode("utf-8", "replace")[:160]
    except OSError as error:
        record["error"] = f"{type(error).__name__}: {error}"
    finally:
        connection.close()
    return record


class ServePhaseTests(unittest.TestCase):
    """The end of the replay is not the end of the run - and a stop still is."""

    def test_the_transport_keeps_answering_after_the_drive_phase_returns(self):
        directory = Path(tempfile.mkdtemp(prefix="demo-serve-"))
        seen: dict = {}
        done = threading.Event()
        original = demo.drive

        def poll(port: int) -> None:
            """Probe the port every 250 ms until the run itself is over."""

            probes = []
            while not done.is_set():
                probes.append(fetch(port))
                done.wait(0.25)
            seen["probes"] = probes

        async def watching(port, args, session, media_report, stop):
            state = await original(port, args, session, media_report, stop)
            # Everything below happens *after* the replay's terminal status was
            # seen: this is the instant the defect used to shut the server down.
            seen["stop_at_replay_end"] = stop.is_set()
            seen["replay_ended_at"] = time.monotonic()
            seen["port"] = port
            poller = threading.Thread(target=poll, args=(port,), daemon=True)
            poller.start()
            seen["poller"] = poller
            return state

        code = None
        demo.drive = watching
        try:
            code = demo.main(arguments(directory, ["--serve-seconds", f"{SERVE_SECONDS:g}"]))
            seen["returned_at"] = time.monotonic()
        finally:
            demo.drive = original
            done.set()
            if "poller" in seen:
                seen["poller"].join(timeout=5)
            shutil.rmtree(directory, ignore_errors=True)

        self.assertEqual(code, 0)
        self.assertIn("replay_ended_at", seen, "the run never reached the end of the replay")
        self.assertFalse(
            seen["stop_at_replay_end"],
            "the drive phase signalled the run's stop event; that is the defect, verbatim",
        )
        served = [
            probe
            for probe in seen.get("probes", [])
            if probe.get("page") == 200 and probe.get("state") == 200
        ]
        self.assertGreaterEqual(
            len(served),
            2,
            f"the port answered {len(served)} time(s) after the replay: "
            f"{seen.get('probes', [])[:3]}",
        )
        span = max(probe["at"] for probe in served) - seen["replay_ended_at"]
        self.assertGreaterEqual(
            span,
            SERVE_SECONDS - 1.0,
            f"the port stopped answering {span:.2f}s into a {SERVE_SECONDS:g}s window",
        )
        self.assertGreaterEqual(
            seen["returned_at"] - seen["replay_ended_at"],
            SERVE_SECONDS - 0.5,
            "the run returned without serving its stay-open window",
        )

    def test_the_run_ends_when_the_operator_stops_it(self):
        """``--browser`` waits for Ctrl-C: a dead stop path would never return."""

        directory = Path(tempfile.mkdtemp(prefix="demo-browser-"))
        seen: dict = {}
        original = demo.drive

        async def stopping(port, args, session, media_report, stop):
            state = await original(port, args, session, media_report, stop)
            fetch_now = fetch(port)
            seen["served_at_stop"] = (fetch_now.get("page"), fetch_now.get("state"))
            seen["stop_before_request"] = stop.is_set()
            seen["replay_ended_at"] = time.monotonic()
            # Exactly what `request_stop` does for Ctrl-C and SIGTERM, from
            # another thread and half a second into the unbounded wait.
            loop = asyncio.get_running_loop()
            timer = threading.Timer(0.5, lambda: loop.call_soon_threadsafe(stop.set))
            timer.daemon = True
            timer.start()
            return state

        def run() -> None:
            try:
                seen["code"] = demo.main(arguments(directory, ["--browser"]))
            except BaseException as error:  # pragma: no cover - reported, not raised
                seen["error"] = f"{type(error).__name__}: {error}"

        demo.drive = stopping
        runner = threading.Thread(target=run, daemon=True)
        try:
            runner.start()
            runner.join(timeout=30)
            seen["ended_at"] = time.monotonic()
        finally:
            demo.drive = original
            shutil.rmtree(directory, ignore_errors=True)

        self.assertFalse(
            runner.is_alive(),
            "the run never ended: the stay-open loop ignored the stop request",
        )
        self.assertNotIn("error", seen)
        self.assertEqual(seen.get("code"), 0)
        self.assertFalse(seen.get("stop_before_request"), "the replay ended the run early")
        self.assertEqual(
            seen.get("served_at_stop"),
            (200, 200),
            "the page and its state endpoint were not up when the stop was requested",
        )
        self.assertLess(
            seen["ended_at"] - seen["replay_ended_at"],
            20.0,
            "the run took longer than 20s to end after the stop request",
        )


if __name__ == "__main__":
    unittest.main()
