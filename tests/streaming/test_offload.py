"""Tests for consumer-side task offloading."""

import time
import unittest
from queue import Full
from threading import Event, Lock

from nova2026.streaming.offload import TaskOffloader


class SubmitTests(unittest.TestCase):
    """Check that submitting never waits for the handler."""

    def test_submit_returns_before_the_handler_finishes(self) -> None:
        started = Event()
        release = Event()

        def handler(item):
            started.set()
            release.wait(5)
            return item

        offloader = TaskOffloader(handler, workers=1, capacity=4)
        try:
            begin = time.monotonic()
            accepted = [offloader.submit(index) for index in range(3)]
            elapsed = time.monotonic() - begin
        finally:
            release.set()
            offloader.close(drain=True, timeout=5)

        self.assertTrue(all(accepted))
        self.assertLess(elapsed, 0.1)
        self.assertEqual(offloader.completed, 3)

    def test_every_item_runs_once_in_order_with_one_worker(self) -> None:
        seen = []
        lock = Lock()

        def handler(item):
            with lock:
                seen.append(item)

        # Capacity must cover the burst, otherwise drop_oldest discards items.
        offloader = TaskOffloader(handler, workers=1, capacity=128)
        for index in range(100):
            offloader.submit(index)
        offloader.close(drain=True, timeout=5)

        self.assertEqual(seen, list(range(100)))
        self.assertEqual(offloader.submitted, 100)
        self.assertEqual(offloader.completed, 100)
        self.assertEqual(offloader.dropped, 0)

    def test_workers_run_concurrently(self) -> None:
        lock = Lock()
        both_running = Event()
        running = 0

        def handler(item):
            nonlocal running
            with lock:
                running += 1
                if running == 2:
                    both_running.set()
            both_running.wait(2)
            return item

        offloader = TaskOffloader(handler, workers=2, capacity=4)
        for index in range(2):
            offloader.submit(index)
        offloader.close(drain=True, timeout=5)

        self.assertTrue(both_running.is_set())
        self.assertEqual(offloader.completed, 2)
        self.assertEqual(offloader.failed, 0)


class OverflowTests(unittest.TestCase):
    """Check the backpressure policies against a blocked worker."""

    def _blocked(self, overflow: str):
        release = Event()
        started = Event()
        seen = []
        lock = Lock()

        def handler(item):
            if item == 0:
                started.set()
                release.wait(5)
            with lock:
                seen.append(item)

        offloader = TaskOffloader(handler, workers=1, capacity=1, overflow=overflow)
        offloader.submit(0)
        self.assertTrue(started.wait(2))
        return offloader, release, seen

    def test_drop_oldest_keeps_the_freshest_item(self) -> None:
        offloader, release, seen = self._blocked("drop_oldest")
        try:
            offloader.submit(1)
            offloader.submit(2)
        finally:
            release.set()
            offloader.close(drain=True, timeout=5)

        self.assertEqual(seen, [0, 2])
        self.assertEqual(offloader.dropped, 1)

    def test_drop_newest_keeps_the_oldest_item(self) -> None:
        offloader, release, seen = self._blocked("drop_newest")
        try:
            offloader.submit(1)
            accepted = offloader.submit(2)
        finally:
            release.set()
            offloader.close(drain=True, timeout=5)

        self.assertFalse(accepted)
        self.assertEqual(seen, [0, 1])
        self.assertEqual(offloader.dropped, 1)

    def test_raise_propagates_full(self) -> None:
        offloader, release, _ = self._blocked("raise")
        try:
            offloader.submit(1)
            with self.assertRaises(Full):
                offloader.submit(2)
        finally:
            release.set()
            offloader.close(drain=True, timeout=5)

    def test_submit_after_close_is_dropped(self) -> None:
        offloader = TaskOffloader(lambda item: item, workers=1)
        offloader.close()

        self.assertFalse(offloader.submit(1))
        self.assertEqual(offloader.dropped, 1)


class FailureTests(unittest.TestCase):
    """Check error transport and lifecycle limits."""

    def test_handler_error_is_transported(self) -> None:
        def handler(item):
            if item == 1:
                raise ValueError("boom")
            return item

        offloader = TaskOffloader(handler, workers=1, capacity=4)
        for index in range(3):
            offloader.submit(index)
        offloader.close(drain=True, timeout=5)

        self.assertEqual(offloader.completed, 2)
        self.assertEqual(offloader.failed, 1)
        with self.assertRaisesRegex(ValueError, "boom"):
            offloader.raise_error()

    def test_result_and_error_callbacks_run(self) -> None:
        results = []
        errors = []
        lock = Lock()

        def handler(item):
            if item == 1:
                raise ValueError("boom")
            return item * 10

        def on_result(value):
            with lock:
                results.append(value)

        def on_error(error):
            with lock:
                errors.append(type(error).__name__)

        offloader = TaskOffloader(
            handler,
            workers=1,
            capacity=4,
            on_result=on_result,
            on_error=on_error,
        )
        for index in range(3):
            offloader.submit(index)
        offloader.close(drain=True, timeout=5)

        self.assertEqual(sorted(results), [0, 20])
        self.assertEqual(errors, ["ValueError"])

    def test_close_reports_a_stuck_handler(self) -> None:
        release = Event()
        offloader = TaskOffloader(lambda item: release.wait(10), workers=1)
        offloader.submit(0)

        try:
            with self.assertRaisesRegex(RuntimeError, "did not stop"):
                offloader.close(timeout=0.2)
        finally:
            release.set()
            time.sleep(0.05)

    def test_invalid_arguments(self) -> None:
        cases = (
            {"handler": None},
            {"handler": len, "workers": 0},
            {"handler": len, "workers": True},
            {"handler": len, "capacity": 0},
            {"handler": len, "overflow": "keep_all"},
        )
        for options in cases:
            with self.subTest(options=options):
                values = dict(options)
                handler = values.pop("handler")
                with self.assertRaises(ValueError):
                    TaskOffloader(handler, **values)


if __name__ == "__main__":
    unittest.main()
