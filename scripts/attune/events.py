"""Cue epochs and JSON markers in LSL source-clock time."""
from dataclasses import asdict, dataclass
from threading import RLock
import json
import math


@dataclass(frozen=True)
class Epoch:
    t_start: float
    t_end: float
    trial_id: str
    cued_side: str
    condition: str
    block_id: str


class EpochLedger:
    def __init__(self, mask_seconds=.5):
        if not math.isfinite(mask_seconds) or mask_seconds < 0:
            raise ValueError('Invalid cue boundary mask.')
        self.mask = mask_seconds
        self.epochs = []
        self.lock = RLock()

    def add(self, t_start, t_end, trial_id, cued_side, condition, block_id):
        if (not all(math.isfinite(t) for t in (t_start, t_end)) or t_end <= t_start
                or cued_side not in ('A', 'B') or condition not in ('dichotic', 'diotic')
                or not all(isinstance(s, str) and s for s in (trial_id, block_id))):
            raise ValueError('Invalid cue epoch.')
        epoch = Epoch(t_start, t_end, trial_id, cued_side, condition, block_id)
        with self.lock:
            if self.epochs and t_start < self.epochs[-1].t_end:
                raise ValueError('Cue epochs overlap or regress.')
            self.epochs.append(epoch)
        return epoch

    def for_window(self, start, end):
        if not all(math.isfinite(t) for t in (start, end)) or end < start:
            raise ValueError('Invalid evidence interval.')
        with self.lock:
            return next((e for e in self.epochs
                         if start > e.t_start+self.mask and end < e.t_end-self.mask), None)


class Markers:
    def __init__(self, source_id='attune-markers'):
        from mne_lsl.lsl import StreamInfo, StreamOutlet
        self.outlet = StreamOutlet(StreamInfo('AAD_Markers', 'Markers', 1, 0., 'string', source_id))

    def push(self, event, timestamp):
        if not math.isfinite(timestamp):
            raise ValueError('Marker requires finite LSL time.')
        self.outlet.push_sample([json.dumps(asdict(event) if isinstance(event, Epoch) else event,
                                          allow_nan=False)], timestamp=timestamp)


class MarkerReceiver:
    """Bounded nonblocking cue ingestion; labels never enter model features."""
    def __init__(self, name='AAD_Markers', *, ledger=None):
        from mne_lsl.lsl import resolve_streams, StreamInlet
        sources = resolve_streams(name=name, timeout=1.)
        if len(sources) != 1:
            raise ValueError('Exactly one named marker stream is required.')
        self.inlet = StreamInlet(sources[0], max_buffered=30., processing_flags=['clocksync'])
        self.inlet.open_stream(timeout=1.)
        self.ledger = ledger if ledger is not None else EpochLedger()

    def poll(self):
        rows, _ = self.inlet.pull_chunk(timeout=0., max_samples=128)
        for row in rows:
            event = json.loads(row[0])
            if 'cued_side' in event:
                self.ledger.add(**{key: event[key] for key in Epoch.__dataclass_fields__})

    def close(self):
        self.inlet.close_stream()
