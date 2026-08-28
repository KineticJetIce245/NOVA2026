"""Build the resting-baseline checkpoint: ``outputs/RS_Beg_EO_128Hz_AttUPipeline.pt``.

Preprocessing
-------------
Identical to the PVT build (see ``cogbci_pvt.py``): every recording passes
through :class:`AttUPipeline` (``pipelines.py``), i.e. the operator
P = K o B o R o B o N of the engagement-z-scoring design doc
(``documents/engagement_zscoring.pdf``, Eq. 1): 60 Hz notch (10 Hz width) ->
4-20 Hz zero-phase Butterworth band-pass -> resample 500 -> 128 Hz ->
4-20 Hz band-pass again -> per-channel centre-scale-clip.

Baseline windows
----------------
The eyes-open pre-task rest run (RS_Beg_EO) of each (subject, session) is
trimmed 1 s (128 samples @ 128 Hz) at each end to remove filter transients,
then tiled into windows of exactly the length and sample count of a trial
window (T = 256 samples = 2000 ms) with hop H = T/2 = 128 samples (50 %
overlap), per Eq. 4 of the design doc.  The window count per recording is
W_r = floor((n - 2*128 - 256)/128) + 1, where n is the resampled length in
samples; the last partial window is discarded.

Checkpoint schema (dict returned by ``torch.load``)
---------------------------------------------------
data            list of W float64 arrays, one per window, in recording order
                    (all windows of a recording are contiguous); each entry has
                    shape (62, 256) and holds EEG in microvolts (the
                    ``permutate`` callback multiplies the raw data by 1e6);
                    channel order is EEG_CHANNELS (62 channels)
metadata        (W, 2) object array    columns [subject, session];
                    subject like "sub-01", session like "ses-S1";
                    row i corresponds to data[i]
channel_names   list[str] of 62 channel names (EEG_CHANNELS)
pipeline        str                    "AttUPipeline"
sample_rate_hz  int                    128
window_length_ms int                   2000
data_type       str                    "RS_Beg_EO"
"""

import mne
import numpy as np
from pipelines import AttUPipeline
from save_pt import save_chkpt

from nova2026.config import DATA_DIR, SAMPLE_RATE, SAMPLE_SIZE, WINDOW_SIZE
from nova2026.data.eeg import Loader

OUTPUT_DIR = DATA_DIR / "COG-BCI" / "outputs"
DIR_MASK = (DATA_DIR / "COG-BCI").parts
EEG_CHANNELS = [
    "Fp1",
    "Fz",
    "F3",
    "F7",
    "FT9",
    "FC5",
    "FC1",
    "C3",
    "T7",
    "CP5",
    "CP1",
    "Pz",
    "P3",
    "P7",
    "O1",
    "Oz",
    "O2",
    "P4",
    "P8",
    "TP10",
    "CP6",
    "CP2",
    "FCz",
    "C4",
    "T8",
    "FT10",
    "FC6",
    "FC2",
    "F4",
    "F8",
    "Fp2",
    "AF7",
    "AF3",
    "AFz",
    "F1",
    "F5",
    "FT7",
    "FC3",
    "C1",
    "C5",
    "TP7",
    "CP3",
    "P1",
    "P5",
    "PO7",
    "PO3",
    "POz",
    "PO4",
    "PO8",
    "P6",
    "P2",
    "CPz",
    "CP4",
    "TP8",
    "C6",
    "C2",
    "FC4",
    "FT8",
    "F6",
    "AF8",
    "AF4",
    "F2",
]

# Window length and hop are derived from config so the baseline cannot
# silently desynchronise from the trial tensor (design doc, implementation map).
HOP_SAMPLES = SAMPLE_SIZE // 2  # H = T/2 = 128 samples -> 50 % overlap
TRIM_SAMPLES = SAMPLE_RATE  # 1 s at each end (zero-phase filter transient)
MIN_WINDOWS_WARN = 30  # below this, median/MAD baseline estimates get noisy


def tagging_cogbci(datarf: Loader.DataFileRef) -> list[str]:
    # Raw tags, e.g. ["sub-01", "ses-S1"] -- identical convention to
    # cogbci_pvt.py so the two checkpoints can be joined on (subject, session).
    return [t for t in datarf.path.parts[:-2] if t not in DIR_MASK]


def data_integrity_check(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    if not np.isclose(raw.info["sfreq"], SAMPLE_RATE):
        raise ValueError(
            f"Pipeline produced {raw.info['sfreq']} Hz; expected {SAMPLE_RATE} Hz."
        )

    missing_channels = sorted(set(EEG_CHANNELS) - set(raw.ch_names))
    if missing_channels:
        raise ValueError(f"Recording is missing EEG channels: {missing_channels}.")
    return raw


def pick_channels(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    raw.pick(EEG_CHANNELS)
    return raw


# create a new loader
loader = Loader("COG-BCI", root=DATA_DIR)
data_type = "RS_Beg_EO"
# search for files named as RS_Beg/End_EC/EO
loader.search(data_type, "name")
# refine for files with .set extension
loader.refine(lambda datarf: datarf.path.suffix == ".set")
# tag the files based on "sub" and "ses"
loader.tag(tagging_cogbci)

if not loader.query_cache:
    raise ValueError(f"No '{data_type}' recordings found under {DATA_DIR / 'COG-BCI'}.")

# load the data into TaggedData
loader.load(mode="eeglab")

# run the pipeline
pipeline = AttUPipeline()
loader.run_pipe(pipeline)

loader.run(data_integrity_check)
loader.run(pick_channels)

meta_list, data_list = loader.slice_const_interval(
    eegchannels=EEG_CHANNELS,
    sample_size=SAMPLE_SIZE,
    trim=(TRIM_SAMPLES, TRIM_SAMPLES),
    hop=HOP_SAMPLES,
    permutate=lambda array: array * 1e6,
)

# Here we can assume that tags are in the right format
meta_list = [(tags[0], tags[1]) for tags in meta_list]

metadata = np.asarray(meta_list, dtype=object)  # (W, 2), one row per window

checkpoint = {
    "data": data_list,  # list[W] of (62, 256) float64, microvolts, per window
    "metadata": metadata,
    "channel_names": list(EEG_CHANNELS),
    "pipeline": type(pipeline).__name__,
    "sample_rate_hz": SAMPLE_RATE,
    "window_length_ms": WINDOW_SIZE,
    "data_type": data_type,
}
save_chkpt(checkpoint, OUTPUT_DIR)
