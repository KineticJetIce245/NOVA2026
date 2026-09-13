"""Display metadata for exactly two inputs; no acquisition or audio processing."""
import re


def validate_sources(sources=None):
    if sources is None:
        sources = [dict(id=key, label=f'Audio Source {key}',
                        input_type='flexible', reference=None) for key in ('A', 'B')]
    if not isinstance(sources, (list, tuple)) or len(sources) != 2:
        raise ValueError('exactly two audio sources are required')
    normalized = []
    for item in sources:
        if not isinstance(item, dict) or item.get('id') not in ('A', 'B'):
            raise ValueError('source IDs must be A and B')
        result = {'id': item['id']}
        for field in ('label', 'input_type'):
            value = item.get(field)
            if not isinstance(value, str) or not value.strip() or len(value) > 128:
                raise ValueError('source display text must be nonempty and bounded')
            result[field] = value
        reference = item.get('reference')
        if reference is not None and (not isinstance(reference, str) or
                re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', reference) is None):
            raise ValueError('reference must be a public opaque identifier, not a URL or path')
        result['reference'] = reference
        normalized.append(result)
    if {item['id'] for item in normalized} != {'A', 'B'}:
        raise ValueError('source IDs must be distinct A and B')
    return sorted(normalized, key=lambda item: item['id'])
