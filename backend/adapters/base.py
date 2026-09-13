from typing import Protocol
from threading import Event


class Producer(Protocol):
    def run(self, publish, stop: Event) -> None:
        """Publish (type, session-relative timestamp, source, JSON payload).

        Run outside request handlers. Return promptly when stop is set.
        Only explicit display EEG should cross this boundary, never raw acquisition.
        """
        ...
