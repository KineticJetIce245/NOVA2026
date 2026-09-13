"""Serialize result records into the existing session-scoped publisher callback."""
import math
from numbers import Real
from backend.app.protocol import make_packet
from backend.app.prediction import PredictionResult
from .contracts import AttentionResult, VigilanceResult, SyncResult, EEGDisplayResult, PublishCallback, SignalQualityResult, GainResult


def number(value, *, minimum=None, maximum=None):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError('measurement must be a real scalar or None')
    value = float(value)
    if not math.isfinite(value) or (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        raise ValueError('measurement outside valid range')
    return value


def encode_result(result):
    """Return (type, timestamp, source, detached JSON payload), without publishing."""
    if isinstance(result, PredictionResult):
        timestamp = result.timestamp if result.timestamp is not None else 0.0
        packet = make_packet('prediction', timestamp, 0, result.provider_id, 'validation', result.to_dict())
        if type(result.metadata.get('simulated')) is bool:
            packet['payload']['simulated'] = result.metadata['simulated']
        return 'prediction', timestamp, result.provider_id, packet['payload']
    if type(result.simulated) is not bool:
        raise ValueError('simulated must be boolean')
    payload = {'simulated': result.simulated}
    if isinstance(result, AttentionResult):
        if result.decision not in ('A', 'B', 'uncertain', 'unavailable'):
            raise ValueError('invalid attention decision')
        kind = 'attention'
        payload.update(decision=result.decision, attended=result.decision if result.decision in ('A', 'B') else None,
                       correlation_a=number(result.correlation_a, minimum=-1, maximum=1),
                       correlation_b=number(result.correlation_b, minimum=-1, maximum=1))
    elif isinstance(result, SignalQualityResult):
        if result.artifact is not None and type(result.artifact) is not bool:
            raise ValueError('artifact must be boolean or None')
        channels = None if result.bad_channels is None else list(result.bad_channels)
        if channels is not None and (isinstance(result.bad_channels, str) or
                                    any(not isinstance(c, str) or not c for c in channels)):
            raise ValueError('bad_channels must be labels')
        kind = 'signal_quality'
        payload.update(quality=None, artifact=result.artifact, bad_channels=channels)
    elif isinstance(result, GainResult):
        kind = 'gain'
        payload.update(a_db=number(result.a_db, minimum=-20, maximum=0),
                       b_db=number(result.b_db, minimum=-20, maximum=0))
    elif isinstance(result, VigilanceResult):
        if result.metric not in ('vigilance', 'lapse_probability'):
            raise ValueError('unsupported vigilance metric')
        kind = 'vigilance'
        value = number(result.value, minimum=0, maximum=1)
        payload.update(metric=result.metric, score=value if result.metric == 'vigilance' else None,
                       lapse_score=value if result.metric == 'lapse_probability' else None)
    elif isinstance(result, SyncResult):
        if not isinstance(result.status, str) or not 0 < len(result.status) <= 128:
            raise ValueError('invalid sync status')
        if result.drift_warning is not None and type(result.drift_warning) is not bool:
            raise ValueError('drift_warning must be bool or None')
        if result.clock_domain is not None and result.clock_domain != 'local_lsl':
            raise ValueError('unsupported audio clock domain')
        payload.update(clock_domain=result.clock_domain,
                       max_block_timing_error_seconds=number(result.max_block_timing_error_seconds, minimum=0),
                       estimated_audio_drift_ppm=number(result.estimated_audio_drift_ppm))
        kind = 'sync'
        payload.update(status=result.status, offset_ms=number(result.offset_ms), drift_warning=result.drift_warning)
    elif isinstance(result, EEGDisplayResult):
        kind = 'eeg_display'
        channels = list(result.channels) if result.channels is not None else None
        if channels is not None and (isinstance(result.channels, str) or not channels or
                                     any(not isinstance(c, str) or not c for c in channels)):
            raise ValueError('invalid display channel labels')
        samples = None
        if result.samples is not None:
            samples = [[number(v) for v in row] for row in result.samples]
            if channels is None or len(samples) != len(channels) or not samples or any(
                    len(row) != len(samples[0]) or any(v is None for v in row) for row in samples):
                raise ValueError('display samples must be rectangular channel-major data')
        rate = number(result.sample_rate, minimum=0)
        if rate == 0:
            raise ValueError('display sample rate must be positive')
        payload.update(channels=channels, samples=samples, sample_rate=rate)
    else:
        raise TypeError('unsupported result record')
    timestamp = number(result.timestamp, minimum=0, maximum=9007199254740991)
    if timestamp is None or result.source == 'server':
        raise ValueError('timestamp required; server source is reserved')
    # Validate/detach using the same envelope rules; real sequence/session are
    # assigned only by Sessions/Publisher, not by this temporary validation packet.
    packet = make_packet(kind, timestamp, 0, result.source, 'validation', payload)
    return kind, timestamp, result.source, packet['payload']


class ResultAdapter:
    def __init__(self, publish: PublishCallback):
        self._publish = publish

    def publish(self, result):
        """Call from Producer.run with its session-scoped publish callback."""
        return self._publish(*encode_result(result))


class ResultProducer:
    """Bridge a coordinator function run_results(adapter, stop) into Sessions.

    The coordinator owns source/algorithm construction and cleanup in try/finally.
    It must honor stop and use bounded source reads. No threads are created here.
    """
    def __init__(self, run_results, *, simulated=False):
        if not callable(run_results) or type(simulated) is not bool:
            raise ValueError('callable coordinator and boolean simulated required')
        self._run_results = run_results
        self.simulated = simulated

    def run(self, publish, stop):
        if not stop.is_set():
            self._run_results(ResultAdapter(publish), stop)
