import mne

from nova2026.config import DATA_DIR
from nova2026.data.eeg import Loader

loader = Loader("COG-BCI", root=DATA_DIR)
loader.search("PVT", "name")
loader.refine(lambda datarf: datarf.path.suffix == ".set")
print(loader.query_cache)
