"""Constants for the PVT visual-detection experiment.

The task predicts, from the *post*-stimulus EEG only, whether the subject
registered the stimulus. It complements ``scripts/dataproc/cogbci_pvt.py``,
which predicts a lapse from the 2 s of signal *before* the stimulus.

The occipital channels are listed literally rather than imported from
``cogbci_pvt``: that module runs the whole checkpoint build at import time, so
importing it for a constant would load all 75 recordings as a side effect.
"""

from pathlib import Path

from nova2026.config import DATA_DIR

#: Reconstruction order of the COG-BCI cap (62 EEG channels), copied from
#: ``scripts/dataproc/cogbci_pvt.py``. ``build_dataset`` verifies its channel set
#: against the loaded recording, so a divergence fails loudly instead of
#: silently reordering channels.
EEG_CHANNELS = (
    "Fp1", "Fz", "F3", "F7", "FT9", "FC5", "FC1", "C3", "T7", "CP5",
    "CP1", "Pz", "P3", "P7", "O1", "Oz", "O2", "P4", "P8", "TP10",
    "CP6", "CP2", "FCz", "C4", "T8", "FT10", "FC6", "FC2", "F4", "F8",
    "Fp2", "AF7", "AF3", "AFz", "F1", "F5", "FT7", "FC3", "C1", "C5",
    "TP7", "CP3", "P1", "P5", "PO7", "PO3", "POz", "PO4", "PO8", "P6",
    "P2", "CPz", "CP4", "TP8", "C6", "C2", "FC4", "FT8", "F6", "AF8",
    "AF4", "F2",
)  # fmt: skip

#: ``occipital`` is the minimal visual set (8 electrodes at the back of the
#: head). ``posterior`` widens it to 17 by adding the surrounding ring --
#: lateral-occipital (PO7/PO8), the parietal-occipital row (P7/P5/P3/P1/Pz/
#: P2/P4/P6/P8) and PO3/POz/PO4 -- on the argument that a visual response is
#: spatially broader than the four midline sites, and that more electrodes also
#: give a spatial filter more to work with. ``all`` is the whole cap, used as a
#: control: if 17 electrodes do not beat 8, adding more sensors is not the
#: bottleneck. All are stored in cap order.
CHANNEL_SETS = {
    "occipital": ["O1", "Oz", "O2", "PO7", "PO3", "POz", "PO4", "PO8"],
    "posterior": [
        "Pz", "P3", "P7", "O1", "Oz", "O2", "P4", "P8", "P1", "P5",
        "PO7", "PO3", "POz", "PO4", "PO8", "P6", "P2",
    ],  # fmt: skip
    "all": list(EEG_CHANNELS),
}

#: Output directory and data-type tag of the PVT-visual checkpoints.
VIS_DIR = DATA_DIR / "COG-BCI" / "outputs"
VIS_DATA_TYPE = "PVT_VIS"


def channels_for(name: str) -> list[str]:
    """Named channel set, verified against :data:`EEG_CHANNELS` and cap-ordered."""
    try:
        requested = CHANNEL_SETS[name]
    except KeyError:
        raise ValueError(
            f"Unknown channel set {name!r}; expected one of {sorted(CHANNEL_SETS)}"
        ) from None
    missing = [ch for ch in requested if ch not in EEG_CHANNELS]
    if missing:
        raise ValueError(f"Channel set {name!r} has unknown channels: {missing}")
    if len(set(requested)) != len(requested):
        raise ValueError(f"Channel set {name!r} lists a channel twice.")
    return [ch for ch in EEG_CHANNELS if ch in set(requested)]


def stem(channel_set: str, lo_ms: int, hi_ms: int) -> str:
    """Descriptive stem for the epoch geometry, e.g. ``occipital_m50_500ms``.

    A negative pre-stimulus bound becomes ``m50`` so the stem is a safe
    filename fragment on every platform.
    """
    prefix = f"m{abs(lo_ms)}" if lo_ms < 0 else str(lo_ms)
    return f"{channel_set}_{prefix}_{hi_ms}ms"


def output_path(data_type: str, pipeline_name: str, sample_rate: float) -> Path:
    """Path ``save_chkpt`` writes for this geometry."""
    return VIS_DIR / f"{data_type}_{sample_rate}Hz_{pipeline_name}.pt"
