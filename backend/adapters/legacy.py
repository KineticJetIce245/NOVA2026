"""Convert existing output shapes without importing/running scientific code."""
from collections.abc import Mapping
from numbers import Integral
from .contracts import AttentionResult, VigilanceResult
from .results import encode_result


def aad_result(attended, correlations, *, timestamp, source='nova-aad', simulated=False):
    """Existing mixer index 0=A, 1=B. None=unavailable; 'uncertain' is explicit."""
    if isinstance(attended, Integral) and not isinstance(attended, bool) and attended in (0, 1):
        decision = ('A', 'B')[int(attended)]
    elif attended is None:
        decision = 'unavailable'
    elif isinstance(attended, str) and attended == 'uncertain':
        decision = 'uncertain'
    else:
        raise ValueError('expected mixer index 0/1, uncertain, or None')
    a, b = (None, None) if correlations is None else correlations
    result = AttentionResult(timestamp, source, decision, a, b, simulated)
    encode_result(result)
    return result


def lapse_result(score, *, timestamp, source='nova-lapse', simulated=False):
    """Accept LapsePredictor.predict's scalar, preserving lapse probability."""
    result = VigilanceResult(timestamp, source, score, 'lapse_probability', simulated)
    encode_result(result)
    return result


def combined_results(decision, *, timestamp, attention_source='nova-aad', vigilance_source='nova-lapse', simulated=False):
    """Accept CombinedDecision or its to_dict() output; explicit session clock."""
    def field(name):
        return decision[name] if isinstance(decision, Mapping) else getattr(decision, name)
    # Validate both before the caller publishes either. These remain independent
    # packets, not an atomic multi-packet transport transaction.
    return (aad_result(field('attended_talker'), field('correlations'), timestamp=timestamp,
                       source=attention_source, simulated=simulated),
            lapse_result(field('lapse_score'), timestamp=timestamp, source=vigilance_source, simulated=simulated))
