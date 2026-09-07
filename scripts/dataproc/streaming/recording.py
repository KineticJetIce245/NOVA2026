"""Append raw chunks and task markers without coupling storage to acquisition."""

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import numpy as np

from .config import StreamConfig
from .run import RunSpec


def array_bytes(data: np.ndarray) -> bytes:
    """Encode numeric arrays without pickle, retaining NaNs and original dtype."""

    stream = BytesIO()
    np.save(stream, data, allow_pickle=False)
    return stream.getvalue()


class RunRecorder:
    """Write one run on its processing thread using a transactional SQLite file.

    Args:
        root: Parent directory for new subject/session/run directories.
        spec: Run identity and explicitly selected input files.
        config: Source units, channel order, and complete processing settings.

    Notes:
        Raw means selected/reordered source samples before voltage conversion,
        filtering, resampling, or spatial correction. The original inlet order
        is recorded separately in source_metadata. Each append is committed.
        No acquisition or storage thread is started by this class.
    """

    def __init__(self, root: Path, spec: RunSpec, config: StreamConfig) -> None:
        """Store a recording plan; open() creates it on the processing thread."""

        self.directory = Path(root) / spec.subject / spec.session / spec.run_id
        self.spec = spec
        self.config = config.updated()
        self.connection: sqlite3.Connection | None = None
        self.metadata: dict = {}

    def open(self, source_metadata: dict, track_windows: bool = False) -> None:
        """Create a new run, refusing overwrite, and snapshot selected inputs.

        Args:
            source_metadata: Observed inlet identity and channel ordering.
            track_windows: Require window events when reconstructing delivery.
                Streamer enables this to retain a callback's exact stopping point.

        Raises:
            FileExistsError: If the run directory already exists.
            OSError: If a selected input cannot be read or storage fails.
        """

        if self.connection is not None:
            raise RuntimeError("The recorder is already open.")

        # Read selections first so a missing input does not consume the run ID.
        snapshots = []

        for name in ("artifact_path", "baseline_path", "model_path"):
            path = getattr(self.spec, name)
            if path is not None:
                source = Path(path).resolve()
                snapshots.append((name, source, source.read_bytes()))

        self.directory.mkdir(parents=True, exist_ok=False)
        inputs = {}

        for name, source, content in snapshots:
            target = self.directory / (name + source.suffix)
            target.write_bytes(content)
            inputs[name] = {
                "source": str(source),
                "snapshot": target.name,
                "sha256": hashlib.sha256(content).hexdigest(),
            }

        self.metadata = {
            "schema": 1,
            "run": self.spec.to_dict(),
            "config": self.config.to_dict(),
            "source_metadata": source_metadata,
            "inputs": inputs,
            "created_utc": datetime.now(UTC).isoformat(),
            "status": "recording",
            "track_windows": track_windows,
        }
        connection = sqlite3.connect(self.directory / "run.sqlite")
        self.connection = connection
        connection.executescript(
            "CREATE TABLE metadata (id INTEGER PRIMARY KEY, value TEXT NOT NULL);"
            "CREATE TABLE chunks (id INTEGER PRIMARY KEY, segment INTEGER, "
            "disposition TEXT, data BLOB, timestamps BLOB);"
            "CREATE TABLE events (id INTEGER PRIMARY KEY, timestamp REAL, "
            "kind TEXT, details TEXT);"
        )
        connection.execute(
            "INSERT INTO metadata VALUES (1, ?)", (json.dumps(self.metadata),)
        )
        connection.commit()

    def write_chunk(
        self,
        data: np.ndarray,
        timestamps: np.ndarray,
        segment: int,
        disposition: str = "received",
    ) -> None:
        """Commit unchanged canonical source samples, including recoverable faults."""

        connection = self._connection()
        connection.execute(
            "INSERT INTO chunks(segment, disposition, data, timestamps) VALUES (?, ?, ?, ?)",
            (segment, disposition, array_bytes(data), array_bytes(timestamps)),
        )
        connection.commit()

    def event(self, timestamp: float, kind: str, details: dict) -> None:
        """Commit a task marker, recovery decision, or failure with its timestamp."""

        if not np.isfinite(timestamp):
            raise ValueError("Event timestamps must be finite.")

        connection = self._connection()
        connection.execute(
            "INSERT INTO events(timestamp, kind, details) VALUES (?, ?, ?)",
            (timestamp, kind, json.dumps(details, allow_nan=False)),
        )
        connection.commit()

    def close(self, status: str, stats: dict, error: str | None = None) -> None:
        """Finalize success/failure metadata and release the database handle."""

        if self.connection is None:
            return

        connection = self.connection
        try:
            self.metadata.update(status=status, stats=stats, error=error)
            connection.execute(
                "UPDATE metadata SET value=? WHERE id=1",
                (json.dumps(self.metadata),),
            )
            connection.commit()
        finally:
            connection.close()
            self.connection = None

    def _connection(self) -> sqlite3.Connection:
        """Return the active database or reject use before open()."""

        if self.connection is None:
            raise RuntimeError("Open the recorder before writing.")

        return self.connection
