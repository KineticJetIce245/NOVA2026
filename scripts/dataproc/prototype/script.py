import torch
from spectual import compute_and_save_psd, compute_engagement_metrics
from nova2026.config import DATA_DIR

DATASET_ROOT = DATA_DIR / "COG-BCI" / "prototype_outputs"

chkpt_eeg = torch.load(
    DATASET_ROOT / "RS_End_EO_data_AttUPipeline.pt", weights_only=False
)
chkpt = torch.load(DATASET_ROOT / "RS_End_EO_psd.pt", weights_only=False)
psd = chkpt["psd"]
freq = chkpt["freq"]
meta = chkpt_eeg["metadata"]
print(psd.shape)
print(freq)
print(meta)
