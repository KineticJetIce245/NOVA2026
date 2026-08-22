import numpy as np
from spectual import compute_and_save_psd, compute_engagement_metrics

from nova2026.config import DATA_DIR

DATASET_ROOT = DATA_DIR / "COG-BCI" / "outputs"
DATASET = "RS_Beg_EC_128Hz_AttUPipeline.pt"
PSD = "RS_Beg_EC_128Hz_AttUPipeline_PSD.pt"

# compute_and_save_psd(DATASET, PSD)

engagements = compute_engagement_metrics(PSD)
log_eng = np.log(engagements)
print(engagements.shape, log_eng.shape)
