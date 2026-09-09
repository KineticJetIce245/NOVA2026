"""Read original run chunks and verify explicitly selected input snapshots."""

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path

import numpy as np

from .config import StreamConfig


class RunReader:
    """Read recorded chunks and metadata without changing an existing run."""

    def __init__(self, directory: Path) -> None:
        """Open an existing run database in SQLite read-only mode."""

        self.directory = Path(directory)
        path = self.directory / "run.sqlite"
        self.connection = sqlite3.connect(
            path.resolve().as_uri() + "?mode=ro", uri=True
        )
        try:
            row = self.connection.execute(
                "SELECT value FROM metadata WHERE id=1"
            ).fetchone()
            if row is None:
                raise ValueError("The run has no metadata.")

            self.metadata = json.loads(row[0])
            if self.metadata["schema"] != 1:
                raise ValueError("Unsupported recording schema.")

            self.config = StreamConfig.from_dict(self.metadata["config"])
        except BaseException:
            self.connection.close()
            raise

    def chunks(self) -> Iterator[tuple[np.ndarray, np.ndarray, str]]:
        """Yield original chunk boundaries in acquisition order, without pickle."""

        rows = self.connection.execute(
            "SELECT data, timestamps, disposition FROM chunks ORDER BY id"
        )

        for data, timestamps, disposition in rows:
            yield (
                np.load(BytesIO(data), allow_pickle=False),
                np.load(BytesIO(timestamps), allow_pickle=False),
                disposition,
            )

    def events(self) -> list[tuple[float, str, dict]]:
        """Return task markers and processing decisions in recorded insertion order."""

        rows = self.connection.execute(
            "SELECT timestamp, kind, details FROM events ORDER BY id"
        )

        return [
            (timestamp, kind, json.loads(details)) for timestamp, kind, details in rows
        ]

    def input_path(self, name: str) -> Path | None:
        """Resolve and verify a snapshotted calibration, baseline, or model file."""

        item = self.metadata["inputs"].get(name)

        if item is None:
            return None
        path = self.directory / item["snapshot"]

        if path.resolve().parent != self.directory.resolve():
            raise ValueError("An input snapshot must be inside its run directory.")

        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("A selected input snapshot no longer matches its hash.")

        return path

    def close(self) -> None:
        """Release the read-only database connection."""

        self.connection.close()
