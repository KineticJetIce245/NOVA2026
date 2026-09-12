"""Build the PVT *visual-detection* checkpoint: EEG aligned to stimulus onset.

Task
----
Classify, from the occipital EEG **100-300 ms after the stimulus**, whether the
subject registered the stimulus. This complements ``cogbci_pvt.py``, which
predicts a lapse from the 2 s of signal *before* the stimulus.

Trial identity comes from the trial structure, not from the raw marker codes:

``stim`` (13)
    Stimulus onset, the alignment point.
``resp`` (14)
    Button press.
``premature`` (12)
    28 trials in the whole dataset. In this log a code-12 marks a premature or
    invalid response, not a new stimulus, so it invalidates the pairing instead
    of opening a trial (``cogbci_pvt.py`` turns it into a trial with RT -1).

Two label schemes are stored side by side. Labels are conditioned only on trial
order and behavioural timestamps, never on the EEG, so the test subject's signal
never informs its own labels.

``vr`` (default, "was it registered at all")
    1 = an epoch aligned to a real stimulus, 0 = an epoch of the same length
    drawn from the same session's event-free silence with a guard against every
    stimulus and response. Balanced by construction, needs no behavioural
    threshold; caveat: the null epochs are drawn uniformly in time while the
    stimulus is a brief flash, so part of the separability is stimulus-locking
    (onset/offset transient, evoked response) rather than a cognitive decision.
``rt`` ("registered in time")
    1 = RT below ``--rt-threshold-ms`` (500 ms, the PVT-lapse convention used
    elsewhere in this repo), 0 = slower or unanswered. Median RT in these
    sessions is 300-390 ms, so the positive class is a minority; ``rt_z`` keeps
    the continuous log-RT for a threshold-free analysis.

Geometry
--------
Each epoch is a fixed ``[--pre-ms, --post-ms]`` window around its alignment
point (``-50`` to ``+500`` ms by default), so the classification window is
chosen at training time without a rebuild and a per-trial baseline correction
can be applied there rather than frozen into the tensor. Epochs are stored in
microvolts as float32, because the raw int16 encoding is the precision limit of
this local copy.

The defaults are ``--band 4 20`` and ``--no-normalize``. Both were chosen from
measurements on the local data: the 4-20 Hz band carries the 100-150 ms evoked
positivity (+3.7 uV, t=+13.6 over 1340 pooled trials) while attenuating the
pre-motor negativity that the 1-40 Hz band lets through, and AttentivU's
per-channel divide-by-8 uV shrinks that response to about 0.5 uV.

Usage
-----
    .venv\\Scripts\\python.exe -m scripts.visual_detect.build_dataset

    # wider posterior montage, no downsampling, wider band, AttentivU scaling
    .venv\\Scripts\\python.exe -m scripts.visual_detect.build_dataset ^
        --channel-set posterior --sample-rate 500 --band 1 40 --normalize
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass, field

import mne
import numpy as np

from nova2026.config import DATA_DIR, SAMPLE_RATE
from nova2026.data.eeg import Loader, save_chkpt

from .device import VIS_DATA_TYPE, VIS_DIR, channels_for, output_path, stem

COG_ROOT = DATA_DIR / "COG-BCI"
DIR_MASK = COG_ROOT.parts

STIM_CODE = "13"
RESPONSE_CODE = "14"
PREMATURE_CODE = "12"

#: EEG sampling rate of the raw PVT recordings, in Hz.
RAW_SAMPLE_RATE = 500.0

IIR_PARAMS = {"order": 4, "ftype": "butter", "output": "sos"}


class VisualPipeline:
    """Occipital-bandpass pipeline for post-stimulus epochs.

    Same operator family as ``scripts/dataproc/pipelines.py:AttUPipeline``
    (notch -> band-pass -> optional resample -> band-pass -> per-channel
    centre/scale/clip) with a configurable band and output rate. The default
    band is wider than AttentivU's 4-20 Hz because a single-trial visual evoked
    response carries energy above 20 Hz that band-pass removes.
    """

    def __init__(
        self,
        l_freq: float = 1.0,
        h_freq: float = 40.0,
        sample_rate: float = float(SAMPLE_RATE),
        notch_freq: float = 60.0,
        notch_widths: float = 10.0,
        normalize: bool = False,
        iir_params: dict | None = None,
        verbose: bool = False,
    ) -> None:
        if l_freq <= 0 or h_freq <= l_freq:
            raise ValueError(f"Invalid band ({l_freq}, {h_freq}) Hz.")
        if sample_rate <= 2 * h_freq:
            raise ValueError(
                f"Sample rate {sample_rate} Hz cannot carry a {h_freq} Hz "
                "low-pass; raise --sample-rate or lower --band."
            )
        if h_freq >= RAW_SAMPLE_RATE / 2.0:
            raise ValueError(
                f"High cutoff {h_freq} Hz is at or above the raw Nyquist "
                f"({RAW_SAMPLE_RATE / 2.0} Hz)."
            )
        self.l_freq = float(l_freq)
        self.h_freq = float(h_freq)
        self.sample_rate = float(sample_rate)
        self.notch_freq = float(notch_freq)
        self.notch_widths = float(notch_widths)
        self.normalize = bool(normalize)
        self.iir_params = dict(iir_params or IIR_PARAMS)
        self.verbose = verbose


def _center_scale_clip(values_v: np.ndarray) -> np.ndarray:
    """AttentivU per-channel centre/scale/clip, estimated on the whole session.

    The location and scale come from the full recording, never from the epoch
    under test, so this cannot leak the test trial into its own input.
    """
    values_uv = values_v * 1e6
    return np.clip((values_uv - values_uv.mean()) / 8.0, -4.0, 4.0) / 1e6


def preprocess(raw: mne.io.BaseRaw, pipe: VisualPipeline) -> mne.io.BaseRaw:
    """Apply :class:`VisualPipeline` to ``raw`` in place and return it.

    ``pipe.normalize`` controls AttentivU's per-channel centre/scale/clip. It is
    off by default for this task: the operator divides by 8 uV, which scales the
    ~4 uV 100-150 ms evoked response in this dataset down to ~0.5 uV and makes
    single-trial detection far harder. Leaving the data in microvolts keeps the
    task honest; per-trial scaling is available at training time instead.
    """
    nyq = raw.info["sfreq"] / 2.0
    if pipe.h_freq >= nyq:
        raise ValueError(
            f"High cutoff {pipe.h_freq} Hz is at or above Nyquist ({nyq} Hz)."
        )
    # MNE only accepts one stop-band per call for an IIR notch, and a stop-band
    # is pointless once the pass-band already ends below the line frequency.
    harmonics = [
        f
        for f in np.arange(pipe.notch_freq, pipe.h_freq, pipe.notch_freq)
        if f > pipe.l_freq
    ]
    for harmonic in harmonics:
        raw.notch_filter(
            freqs=float(harmonic),
            notch_widths=pipe.notch_widths,
            method="iir",
            iir_params=pipe.iir_params,
            verbose=pipe.verbose,
        )
    band = dict(
        l_freq=pipe.l_freq,
        h_freq=pipe.h_freq,
        method="iir",
        iir_params=pipe.iir_params,
        verbose=pipe.verbose,
    )
    raw.filter(**band)
    if not np.isclose(raw.info["sfreq"], pipe.sample_rate):
        raw.resample(pipe.sample_rate, verbose=pipe.verbose)
        raw.filter(**band)  # the antialiasing filter leaves edge transients
    if pipe.normalize:
        raw.apply_function(_center_scale_clip, channel_wise=True, verbose=pipe.verbose)
    return raw


@dataclass
class Trials:
    """Trial timings of one recording, in milliseconds."""

    stim_ms: np.ndarray
    rt_ms: np.ndarray  # NaN when the trial was never answered
    premature_ms: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))

    def __len__(self) -> int:
        return int(self.stim_ms.size)


def read_trials(raw: mne.io.BaseRaw) -> Trials:
    """Pair every stimulus with the next response, in time order.

    A code-12 press is always premature: it fires *before* the stimulus it was
    meant for, so either it replaced the previous trial's response or it stole
    the current trial's press, and the log does not record which. Both trials
    are therefore marked unusable instead of guessing an assignment.
    """
    stim, resp, premature = [], [], []
    for annotation in raw.annotations:
        onset_ms = int(round(float(annotation["onset"]) * 1000.0))
        description = str(annotation["description"])
        if description == STIM_CODE:
            stim.append(onset_ms)
        elif description == RESPONSE_CODE:
            resp.append(onset_ms)
        elif description == PREMATURE_CODE:
            premature.append(onset_ms)

    stim_arr = np.asarray(stim, dtype=np.int64)
    resp_arr = np.asarray(resp, dtype=np.int64)
    premature_arr = np.asarray(premature, dtype=np.int64)
    rt = np.full(stim_arr.shape, np.nan, dtype=np.float64)

    # 1. each stimulus takes the earliest response that follows it
    next_resp = 0
    for i, onset in enumerate(stim_arr):
        while next_resp < resp_arr.size and resp_arr[next_resp] < onset:
            next_resp += 1
        if next_resp >= resp_arr.size:
            break
        rt[i] = float(resp_arr[next_resp] - onset)
        next_resp += 1

    # 2. a premature press makes its own trial and the trial before it
    #    ambiguous: it either replaced that earlier response or stole the
    #    current one, and the log does not say which. Clear both.
    for press in premature_arr:
        after = int(np.searchsorted(stim_arr, press, side="left"))
        for index in (after - 1, after):
            if 0 <= index < rt.size:
                rt[index] = np.nan

    return Trials(stim_arr, rt, premature_arr)


def noise_regions(
    duration_ms: int, forbidden_ms: Iterable[int], guard_ms: int
) -> list[tuple[int, int]]:
    """Event-free ``[start, end)`` regions of a recording, in milliseconds.

    Every forbidden event removes ``guard_ms`` on both sides.
    """
    if guard_ms < 0:
        raise ValueError("guard_ms must not be negative.")
    blocked = sorted(int(t) - guard_ms for t in forbidden_ms)
    freed = sorted(int(t) + guard_ms for t in forbidden_ms)
    regions: list[tuple[int, int]] = []
    cursor = 0
    for start, end in zip(blocked, freed):
        if start > cursor:
            regions.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration_ms:
        regions.append((cursor, duration_ms))
    return [(s, e) for s, e in regions if e > s]


def sample_nulls(
    regions: list[tuple[int, int]],
    n: int,
    window_ms: int,
    rng: np.random.Generator,
    min_gap_ms: int | None = None,
) -> np.ndarray:
    """Draw ``n`` window starts (ms) that stay inside ``regions``.

    A window never straddles a region boundary, so every returned start is at
    least ``window_ms`` clear of the events that carved the regions out.
    """
    if n < 0:
        raise ValueError("n must not be negative.")
    if window_ms <= 0:
        raise ValueError("window_ms must be positive.")
    gap = window_ms if min_gap_ms is None else min_gap_ms
    pool: list[int] = []
    for start, end in regions:
        latest = end - window_ms
        if latest >= start:
            pool.extend(range(start, latest + 1, gap))
    if not pool:
        return np.empty(0, dtype=np.int64)
    candidates = np.asarray(pool, dtype=np.int64)
    rng.shuffle(candidates)
    picked = candidates[:n] if n < candidates.size else candidates
    return np.sort(picked)


def _tag(datarf: Loader.DataFileRef) -> list[str]:
    return [t for t in datarf.path.parts[:-2] if t not in DIR_MASK]


def deterministic_seed(subject: str, session: str) -> int:
    """Stable per-(subject, session) seed.

    ``hash()`` is salted per process for strings, so it would give a different
    null sample on every run and break reproducibility.
    """
    digits = lambda text: int("".join(ch for ch in text if ch.isdigit()) or 0)
    return digits(subject) * 100 + digits(session)


def _require_channels(raw: mne.io.BaseRaw, channels: list[str]) -> mne.io.BaseRaw:
    missing = sorted(set(channels) - set(raw.ch_names))
    if missing:
        raise ValueError(f"Recording is missing requested channels: {missing}")
    return raw


def build(args: argparse.Namespace) -> None:
    channels = channels_for(args.channel_set)
    pipe = VisualPipeline(
        l_freq=args.band[0],
        h_freq=args.band[1],
        sample_rate=args.sample_rate,
        normalize=args.normalize,
    )
    loader = Loader("COG-BCI", root=DATA_DIR)
    loader.search("PVT", "name")
    loader.refine(lambda datarf: datarf.path.suffix == ".set")
    loader.tag(_tag)
    tagged = loader.load(mode="eeglab")

    # 1. preprocess, then keep only the channels of interest
    print(
        f"Preprocessing {len(tagged)} recordings "
        f"({pipe.l_freq:g}-{pipe.h_freq:g} Hz -> {pipe.sample_rate:g} Hz, "
        f"{len(channels)} channels) ..."
    )
    loader.run(lambda raw: preprocess(raw, pipe))
    loader.run(lambda raw: _require_channels(raw, channels))
    loader.run(lambda raw: raw.pick(channels))

    # 2. read the trial structure *after* preprocessing: MNE rescales annotation
    #    onsets when it resamples, so onsets read before the pipeline are in the
    #    raw time base and no longer index the filtered recording
    raw_trials = [read_trials(item.raw) for item in tagged]
    raw_times = [int(item.raw.n_times) for item in tagged]

    per_ms = pipe.sample_rate / 1000.0
    window = int(round((args.post_ms - args.pre_ms) * per_ms))
    low = int(round(args.pre_ms * per_ms))
    high = int(round(args.post_ms * per_ms))
    if window <= 0 or low >= high:
        raise ValueError(f"Empty epoch window [{args.pre_ms}, {args.post_ms}] ms.")

    def to_index(onsets_ms: np.ndarray) -> np.ndarray:
        scaled = np.asarray(onsets_ms, dtype=np.float64) * per_ms
        return np.rint(scaled).astype(np.int64)

    # 3. plan the epochs
    vr_cuts: list[np.ndarray] = []
    vr_labels: list[np.ndarray] = []
    vr_meta: list[tuple] = []
    rt_cuts: list[np.ndarray] = []
    rt_labels: list[np.ndarray] = []
    rt_values: list[np.ndarray] = []
    rt_meta: list[tuple] = []
    n_unanswered = 0
    run = 0

    for item, trials, n_times in zip(tagged, raw_trials, raw_times):
        subject, session = item.tags[0], item.tags[1]
        base = to_index(trials.stim_ms)
        # the stored epoch covers samples [base + low, base + window)
        inside = (base + low >= 0) & (base + window <= n_times)
        answered = np.isfinite(trials.rt_ms)
        n_unanswered += int((inside & ~answered).sum())

        usable = inside & answered
        rt_cuts.append(base[usable])
        rt_values.append(trials.rt_ms[usable])
        rt_labels.append(np.zeros(int(usable.sum()), dtype=np.int64))
        rt_meta.extend([(subject, session, "stim")] * int(usable.sum()))

        if args.null_scheme == "vr":
            keep = inside
            response_ms = trials.stim_ms + np.nan_to_num(trials.rt_ms, nan=0.0)
            forbidden = np.concatenate(
                [trials.stim_ms, trials.premature_ms, response_ms]
            )
            # the run index keeps the silence draw from being identical on
            # every rebuild: a fixed seed would make the null epochs of a
            # session recognisable across checkpoints
            rng = np.random.default_rng(
                args.seed + deterministic_seed(subject, session) + 100_000 * run
            )
            null_ms = sample_nulls(
                noise_regions(int(round(n_times / per_ms)), forbidden, args.guard_ms),
                n=int(keep.sum()),
                window_ms=args.post_ms - args.pre_ms,
                rng=rng,
            )
            run += 1
        else:
            keep = usable
            null_ms = np.empty(0, dtype=np.int64)

        n_stim = int(keep.sum())
        vr_cuts.append(np.concatenate([base[keep], to_index(null_ms)]))
        vr_labels.append(
            np.concatenate(
                [
                    np.ones(n_stim, dtype=np.int64),
                    np.zeros(null_ms.size, dtype=np.int64),
                ]
            )
        )
        vr_meta.extend([(subject, session, "stim")] * n_stim)
        vr_meta.extend([(subject, session, "null")] * int(null_ms.size))

    rt_all = np.concatenate(rt_values) if rt_values else np.empty(0, np.float64)
    rt_labels = [(v < args.rt_threshold_ms).astype(np.int64) for v in rt_values]
    if args.null_scheme == "vr" and not any(c.size for c in vr_cuts):
        raise ValueError(
            "No silence windows could be drawn; lower --guard-ms or use "
            "--null-scheme rt."
        )

    # 4. cut
    def cut(cuts: list[np.ndarray]) -> np.ndarray:
        _, data = loader.slice(
            [c.tolist() for c in cuts],
            eeg_channels=channels,
            sample_size=window,
            cut_at_start=True,
            permutate=lambda w: (w * 1e6).astype(np.float32),
        )
        if not data:
            raise ValueError("No epochs were cut.")
        return np.stack(data, axis=0)

    print("Cutting stimulus epochs (RT label scheme) ...")
    data_rt = cut(rt_cuts)
    print("Cutting stimulus + silence epochs (VR label scheme) ...")
    data_vr = cut(vr_cuts)

    labels = {
        "vr": np.concatenate(vr_labels) if vr_labels else np.empty(0, np.int64),
        "rt": np.concatenate(rt_labels) if rt_labels else np.empty(0, np.int64),
        "rt_ms": rt_all,
        "rt_log_z": _log_z(rt_all),
    }
    metadata = vr_meta
    if data_vr.shape[0] != len(metadata):
        raise ValueError(
            f"Cut {data_vr.shape[0]} epochs for {len(metadata)} planned windows."
        )
    if data_rt.shape[0] != len(rt_meta):
        raise ValueError(
            f"Cut {data_rt.shape[0]} stimulus epochs for {len(rt_meta)} planned."
        )
    if labels["vr"].size != data_vr.shape[0]:
        raise ValueError("VR labels and VR epochs disagree in length.")

    data_type = f"{VIS_DATA_TYPE}_{stem(args.channel_set, args.pre_ms, args.post_ms)}"
    checkpoint = {
        # (n_vr, C, T) stimulus + silence, and (n_rt, C, T) stimulus only
        "data_vr": data_vr,
        "data_rt": data_rt,
        "labels": labels,
        "metadata": metadata,
        "rt_metadata": rt_meta,
        "channel_names": list(channels),
        "sample_rate_hz": pipe.sample_rate,
        "raw_sample_rate_hz": RAW_SAMPLE_RATE,
        "pre_ms": args.pre_ms,
        "post_ms": args.post_ms,
        "band_hz": list(args.band),
        "normalize": pipe.normalize,
        "line_freq_hz": pipe.notch_freq,
        "rt_threshold_ms": args.rt_threshold_ms,
        "guard_ms": args.guard_ms,
        "n_unanswered_dropped": n_unanswered,
        "n_positive_by_scheme": {
            key: int(np.sum(np.asarray(value) == 1))
            for key, value in labels.items()
            if key in ("vr", "rt")
        },
        "pipeline": pipe.__class__.__name__,
        "data_type": data_type,
        "source": "COG-BCI PVT (post-stimulus epochs)",
    }
    save_chkpt(checkpoint, VIS_DIR)
    print(f"Wrote {output_path(data_type, checkpoint['pipeline'], pipe.sample_rate)}")
    print(
        f"  VR: {data_vr.shape[0]} epochs ({labels['vr'].sum()} stimulus / "
        f"{int((labels['vr'] == 0).sum())} silence) | "
        f"RT: {data_rt.shape[0]} trials ({labels['rt'].sum()} fast), "
        f"{n_unanswered} unanswered dropped"
    )


def _log_z(rt_ms: np.ndarray) -> np.ndarray:
    """Log-RT in milliseconds, z-scored over all trials (NaN where unmeasured)."""
    rt = np.asarray(rt_ms, dtype=np.float64)
    out = np.full(rt.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(rt) & (rt > 0)
    if valid.sum() < 2:
        return out
    logged = np.log(rt[valid])
    out[valid] = ((logged - logged.mean()) / logged.std()).astype(np.float32)
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build post-stimulus PVT visual-detection checkpoints."
    )
    parser.add_argument(
        "--channel-set",
        default="occipital",
        help="occipital (8 ch) or posterior (17 ch)",
    )
    parser.add_argument("--pre-ms", type=int, default=-50, help="window start, ms")
    parser.add_argument("--post-ms", type=int, default=500, help="window end, ms")
    parser.add_argument("--band", type=float, nargs=2, default=(4.0, 20.0))
    parser.add_argument("--sample-rate", type=float, default=float(SAMPLE_RATE))
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="apply AttentivU per-channel centre/scale/clip (divides by 8 uV)",
    )
    parser.add_argument(
        "--guard-ms", type=int, default=700, help="event clearance of null windows"
    )
    parser.add_argument("--rt-threshold-ms", type=float, default=500.0)
    parser.add_argument("--null-scheme", choices=("vr", "rt"), default="vr")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.pre_ms >= args.post_ms:
        raise SystemExit("--pre-ms must be smaller than --post-ms.")
    build(args)


if __name__ == "__main__":
    main()
