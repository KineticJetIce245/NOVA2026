"""The KU Leuven auditory contract step 4 freezes: channels, groups, units.

A contract that moves after a decoder is trained invalidates the model:
``AuditoryProcessor.contract`` is compared against a trained model's recorded
contract before a session starts (``scripts/auditory/live.py`` compares the two
dicts outright), so the meanings fixed here are the ones every later measurement
of that model is made against.

Three of them are easy to get wrong by hand, which is why they live in one
importable place instead of in prose:

* ``rep_part{N}_track{M}_dry.wav`` replays ``part{N}_track{M}_dry.wav`` for its
  first 125 s, sample for sample (fact 7 of ``final_connection.md``, re-measured
  by :func:`rep_identity`). A group key that kept the ``rep_`` prefix therefore
  split one story across two groups and let the same 125 s fall on both sides of
  a split. :func:`canonical_story` folds the prefix away; :func:`group_key` is
  the key a leave-one-story split must use.
* ``source_unit_exponent`` is the power of ten of the *source* unit relative to
  volts -- 0 for volts, -6 for microvolts -- the convention of
  ``nova2026.streaming.preprocess.units.unit_scaler`` and of ``Repair``'s
  ``_to_uv = 10 ** (exponent + 6)``. KU Leuven ships microvolts, so the exponent
  for these trials is -6, which makes the endpoint check an identity on the
  recorded values; the ANT rig's volts are 0.
* Candidates are longer than the EEG in every trial, so the audio kept for a
  trial is the prefix lying at or before the last EEG sample
  (:func:`usable_audio_samples`). A window reaching past that has no label and no
  envelope to align against.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATASET = REPO / "datasets" / "AAD-KULeuven"
METADATA_PATH = DATASET / "metadata.json"
CONVERTED = DATASET / "converted"
STIMULI = DATASET / "stimuli"
ENVELOPES = REPO / "datasets" / "audio"
RESULTS = REPO / "results"

# The 20 electrodes the live 24-channel ANT cap shares with the KU Leuven
# 64-channel cap, as (name, KU Leuven column index). The live cap's four extra
# electrodes are frontal-temporal/mastoid and have no KU Leuven column, so a
# 20-channel contract is the only one both recordings can satisfy.
SHARED_CHANNELS = (
    ("Fp1", 0), ("Fp2", 33), ("F7", 6), ("F3", 4), ("Fz", 37), ("F4", 39),
    ("F8", 41), ("T7", 14), ("C3", 12), ("C4", 49), ("T8", 51), ("Cz", 47),
    ("P7", 22), ("P3", 20), ("Pz", 30), ("P4", 57), ("P8", 59), ("Oz", 28),
    ("O1", 26), ("O2", 63),
)
LIVE_ONLY_CHANNELS = ("F9", "F10", "M1", "M2")
# The live cap's own column order, measured from the .cnt files (section 3.11).
LIVE_CAP_CHANNELS = (
    "Fp1", "Fp2", "F9", "F7", "F3", "Fz", "F4", "F8", "F10", "M1", "T7", "C3",
    "C4", "T8", "M2", "Cz", "P7", "P3", "Pz", "P4", "P8", "Oz", "O1", "O2",
)

EXPECTED_SAMPLE_RATE = 128.0
EXPECTED_AUDIO_RATE = 44100.0
EXPECTED_CHANNELS = 64
REPEAT_PREFIX = "rep_"
# Pre-registered before the audit ran: the published schedule's shortest excerpt
# is 124 s and its longest trial 399 s, so a converted trial outside this range
# means the crop is wrong rather than that the range is wrong.
PLAUSIBLE_DURATION_SECONDS = (60.0, 600.0)
# Microvolts are the unit of the converted trials; volts are the ANT rig's.
UNIT_EXPONENT_UV = -6
UNIT_EXPONENT_V = 0


def to_uv_scale(source_unit_exponent):
    """The factor ``Repair`` applies to endpoints, restated for the report."""

    return 10.0 ** (source_unit_exponent + 6)


def canonical_story(name):
    """The story a candidate file carries, with the repeated excerpt folded in.

    ``rep_part1_track1_dry.wav`` and ``part1_track1_dry.wav`` are the same
    recording for the first 125 s, so they are one story and must share one
    group. Only the ``rep_`` prefix is folded: the track identity and the ``_dry``
    provenance stay part of the name.
    """

    text = Path(str(name)).name
    return text[len(REPEAT_PREFIX):] if text.startswith(REPEAT_PREFIX) else text


def canonical_group(group):
    """Canonicalise a stored ``group`` string without reordering its tokens."""

    tokens = [canonical_story(token) for token in str(group).split("|") if token]
    if not tokens:
        raise ValueError("A group must name at least one candidate.")
    return "|".join(tokens)


def group_key(candidates):
    """The frozen group key: the sorted, canonicalised candidate story pair.

    Sorted so the key does not depend on which candidate the subject attended,
    which is what makes it usable as a split key.
    """

    return "|".join(sorted(canonical_story(name) for name in candidates))


def usable_audio_samples(samples, sample_rate, audio_rate):
    """How many candidate samples lie at or before the last EEG sample.

    The EEG spans ``[0, (samples - 1) / sample_rate]`` on the trial clock and the
    candidates start there too, so a candidate sample is inside the trial when
    its index is at most ``(samples - 1) * audio_rate / sample_rate``. The
    converted audio is longer than the EEG in every trial, and the prefix that
    survives is the whole of the ground-truthed signal.
    """

    if samples <= 0 or not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("A trial needs samples at a positive sample rate.")
    if not np.isfinite(audio_rate) or audio_rate <= 0:
        raise ValueError("The candidate rate must be finite and positive.")
    return int(np.floor((samples - 1) * audio_rate / sample_rate)) + 1


def sha256_file(path):
    """Stream a file's hash, so a 35 MB stimulus is never read into memory."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_metadata(path=METADATA_PATH):
    """The committed channel order and provenance, or a clear failure."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def channel_subset_report(names):
    """Check the frozen 20-channel indices against an actual channel order.

    The live cap's own column order is measured data and need not be the shared
    order; what must hold is that its channel *set* is exactly the 20 shared
    electrodes plus the 4 the KU Leuven cap does not have, with no repeats and
    nothing else.
    """

    indices = [index for _, index in SHARED_CHANNELS]
    expected = [name for name, _ in SHARED_CHANNELS]
    observed = [names[index] for index in indices]
    cap = list(LIVE_CAP_CHANNELS)
    return {
        "kuleuven_count": len(names),
        "kuleuven_names": list(names),
        "shared_count": len(expected),
        "shared_names": expected,
        "shared_indices": indices,
        "shared_observed_at_indices": observed,
        "shared_indices_unique": len(set(indices)) == len(indices),
        "shared_matches": observed == expected,
        "live_only": list(LIVE_ONLY_CHANNELS),
        "live_cap_order": cap,
        "live_cap_count": len(cap),
        "live_cap_set_matches": len(cap) == len(set(cap))
        and set(cap) == set(expected) | set(LIVE_ONLY_CHANNELS),
    }


def unit_convention():
    """The unit rule both datasets are read under, with its two instances."""

    return {
        "meaning": "power of ten of the source unit relative to volts",
        "volts": UNIT_EXPONENT_V,
        "microvolts": UNIT_EXPONENT_UV,
        "kuleuven_trials": UNIT_EXPONENT_UV,
        "ant_rig": UNIT_EXPONENT_V,
        "repair_to_uv_at_kuleuven": to_uv_scale(UNIT_EXPONENT_UV),
        "repair_to_uv_at_ant": to_uv_scale(UNIT_EXPONENT_V),
    }


def contract_snapshot(trial_path=None):
    """Capture the live contract values from the code, not from a copy of them.

    Without a converted trial the processor cannot be built (it needs a source
    rate and a channel list), so only the configuration half is captured and the
    report says which half is missing rather than inventing a value.
    """

    from nova2026.auditory.config import MIN_MARGIN, AuditoryConfig

    config = AuditoryConfig()
    snapshot = {
        "auditory_config": config.to_dict(),
        "lag_samples": config.lag_samples,
        "min_margin": MIN_MARGIN,
        "unit_convention": unit_convention(),
        "processor_contract": None,
        "processor_contract_keys": None,
        "source_unit_exponent_default": None,
        "repair_to_uv_default": None,
    }
    if trial_path is None or not Path(trial_path).is_file():
        return snapshot
    from nova2026.auditory.data import load_trial
    from nova2026.auditory.streaming import AuditoryProcessor, stream_config

    settings = stream_config(load_trial(trial_path), config)
    processor = AuditoryProcessor(settings)
    snapshot["processor_contract"] = processor.contract
    snapshot["processor_contract_keys"] = sorted(processor.contract)
    snapshot["source_unit_exponent_default"] = int(settings.source_unit_exponent)
    # The private attribute is the evidence: it is the value the endpoint safety
    # check multiplies by, and no public reader exists for it.
    snapshot["repair_to_uv_default"] = float(vars(processor.repair)["_to_uv"])
    return snapshot


def envelope_snapshot(path=None):
    """The precomputed-envelope record layout, read from a real envelope."""

    path = ENVELOPES / "part1_track1_dry.npz" if path is None else Path(path)
    if not path.is_file():
        return {"path": str(path), "present": False}
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        return {
            "path": str(path),
            "present": True,
            "array_keys": sorted(archive.files),
            "envelope_shape": list(archive["envelope"].shape),
            "envelope_dtype": str(archive["envelope"].dtype),
            "timestamps_dtype": str(archive["timestamps"].dtype),
            "metadata_keys": sorted(metadata),
            "generator_version": metadata["generator_version"],
            "envelope_method": metadata["envelope_method"],
            "sample_rate": metadata["sample_rate"],
            "band": metadata["band"],
            "audio_rate": metadata["audio_rate"],
        }


def rep_identity(stimuli_dir=None):
    """Re-check that every ``rep_`` candidate replays its base, sample for sample.

    Runs on the source wavs rather than on the converted arrays, so it is
    independent evidence for the group rule instead of a restatement of it.
    """

    from nova2026.auditory.audio import read_audio

    root = STIMULI if stimuli_dir is None else Path(stimuli_dir)
    records = []
    for part in (1, 2, 3, 4):
        for track in (1, 2):
            base = root / f"part{part}_track{track}_dry.wav"
            repeat = root / f"rep_part{part}_track{track}_dry.wav"
            if not (base.is_file() and repeat.is_file()):
                records.append({"base": base.name, "rep": repeat.name, "present": False})
                continue
            base_audio, base_rate = read_audio(base)
            repeat_audio, repeat_rate = read_audio(repeat)
            count = min(len(base_audio), len(repeat_audio))
            difference = float(np.max(np.abs(base_audio[:count] - repeat_audio[:count])))
            records.append({
                "base": base.name,
                "rep": repeat.name,
                "present": True,
                "base_seconds": len(base_audio) / base_rate,
                "rep_seconds": len(repeat_audio) / repeat_rate,
                "compared_samples": int(count),
                "max_abs_difference": difference,
                "identical_prefix": difference == 0.0,
                "base_sha256": sha256_file(base),
                "rep_sha256": sha256_file(repeat),
            })
    return records
