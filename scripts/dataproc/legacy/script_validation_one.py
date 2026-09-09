"""Driver: fit resting baselines and score PVT trials with the z-scoring stage.

Reads the checkpoints built by ``cogbci_rest.py`` (RS_Beg_EO) and
``cogbci_pvt.py`` (PVT), fits one baseline per (subject, session) from the
resting windows (design doc Eqs. 8-9), then scores every trial per channel
(Eq. 10) and as a composite score (Eq. 13).  The maths lives in
``engage_z.py``; this script only wires checkpoints to it and prints a
console summary.

Run from the repo root with the venv python:
    .venv\\Scripts\\python.exe scripts\\dataproc\\script.py
"""

import numpy as np
import torch
from engage_z import fit_baselines, lapse_contrast, normalize, normalize_composite

from nova2026.config import DATA_DIR

DATASET_ROOT = DATA_DIR / "COG-BCI" / "outputs"
REST_DATASET = "RS_Beg_EO_128Hz_AttUPipeline.pt"  # baseline: eyes-open rest
PVT_DATASET = "RS_Beg_EC_128Hz_AttUPipeline.pt"  # scored: eyes-closed rest

rest_checkpoint = torch.load(DATASET_ROOT / REST_DATASET, weights_only=False)
pvt_checkpoint = torch.load(DATASET_ROOT / PVT_DATASET, weights_only=False)

baselines = fit_baselines(rest_checkpoint)
print(f"Fitted {len(baselines)} baselines (expected 75).")

z_chan = normalize(pvt_checkpoint, baselines)  # (N, 62) per-channel z, Eq. 10
z_bar = normalize_composite(pvt_checkpoint, baselines)  # (N,) composite, Eq. 13

meta = np.asarray(pvt_checkpoint["metadata"], dtype=object)
print(
    f"Trials: N={len(z_bar)}, z shape {z_chan.shape}, "
    f"Z_bar in [{z_bar.min():.3f}, {z_bar.max():.3f}] "
    f"Z_bar mean {z_bar.mean():.4f}, Z_bar std {z_bar.std():.4f}\n"
)
