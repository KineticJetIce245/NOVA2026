import torch
from nova2026.config import DATA_DIR
from pathlib import Path

ROOT = DATA_DIR / "COG-BCI"
DATASET = ROOT / "PVT_data_window_1.pt"

def load_dataset(path: Path | str):
  checkpoint = torch.load(path, weights_only=False)
  data = checkpoint["data"]
  labels = checkpoint["labels"]
  meta = checkpoint["metadata"]
  sub = meta[:,0]
  reaction_time = meta[:,2].astype(float)
  return data, labels, meta, sub, reaction_time

d, l, m, s, r = load_dataset(DATASET)

print(d.shape)
print(s.shape)