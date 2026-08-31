import numpy as np
import torch
from utils import compute_baseline, log_engagement

from nova2026.config import DATA_DIR

ROOT = DATA_DIR / "COG-BCI" / "outputs"
BASELINE_DATA = ROOT / "RS_Beg_EO_128Hz_AttUPipeline.pt"
TESTING_DATA = ROOT / "RS_Beg_EC_128Hz_AttUPipeline.pt"
# TESTING_DATA = ROOT / "PVT_128Hz_AttUPipeline.pt"

baseline_checkpoint = torch.load(BASELINE_DATA, weights_only=False)
test_checkpoint = torch.load(TESTING_DATA, weights_only=False)

baseline_data = baseline_checkpoint["data"]
baseline_meta = baseline_checkpoint["metadata"]

test_data = test_checkpoint["data"]
test_meta = test_checkpoint["metadata"]


b_meta, b_meds, b_sigs, b_z_avgs, b_sigGs = compute_baseline(
    baseline_data, baseline_meta
)

# t_log_engs is np.ndarray(n trials, n channels)
t_log_engs = log_engagement(test_data)
# b_meds is np.ndarray(n sessions, n channels)
for m in b_meta:  # loop throught all trials
    session_log_engs = t_log_engs[[tag == m for tag in test_meta], :]
    idx = b_meta.index(m)
    # session_log_engs is np.ndarray(n windows, n channels)
    # b_meds[idx] is np.ndarray(n channels)
    # b_sigs[idx] is np.ndarray(n channels)
    t_z_score = (session_log_engs - b_meds[idx]) / b_sigs[idx]
    # t_z_score is np.ndarray(n windows, n channels)
    t_z_bar = np.mean(t_z_score, axis=1) / b_sigGs[idx]
    # t_z_bar is np.ndarray(n windows)
    print(f"Session {m}: {t_z_bar}, mean: {np.mean(t_z_bar)}")
