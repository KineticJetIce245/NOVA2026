# NOVA2026

Auditory-attention training and replay are documented in
[`scripts/auditory/README.md`](scripts/auditory/README.md).

NOVA Buildathon 2026 project: EEG-based attention-lapse detection using a passive BCI approach.

## Folder structure

```
NOVA2026/
├── datasets/     Raw datasets (git-ignored)
├── documents/    Learning material (PDFs) + typst sources
├── references/   Reference PDFs
├── models/       Saved models (git-ignored)
├── scripts/      Development scripts (scratch / experiments)
├── src/          Python code
├── refs.bib      Bibliography (BibTeX)
└── README.md
```

| Folder | Purpose |
| --- | --- |
| `src/` | **Reusable code goes here.** Organize it into subpackages. |
| `scripts/` | **Development / experimental scripts go here.** Anything still in scratch development (data exploration, prototyping, one-off runs) lives in `scripts/`, organized into subfolders per topic. |
| `datasets/` | **Put raw datasets here.** The folder already exists and is the recommended location. Keep raw data intact (`.set`/`.fdt`, `.cnt`, behavioral logs, channel locations). |
| `models/` | **Saved models / checkpoints go here.** This folder is git-ignored so large weights never get committed. |
| `documents/` | Handouts, challenge briefs, and learning notes (PDF), plus their `typst/` sources. |
| `references/` | Reference papers / device documentation (e.g. `attentivU.pdf`). |
| `refs.bib` | Bibliography — see below. |

> Note: `datasets/*` and `models/*` are git-ignored (only their `.gitkeep` is tracked), so large raw data and weights never get committed.

## Setup the environment

> **Before running `uv sync`:** if you want the **CUDA build of PyTorch**, uncomment the
> `[tool.uv.sources]` / `[[tool.uv.index]]` `pytorch-cu132` block at the top of `pyproject.toml`.
> Leave it commented if you want a CPU-only install.

1. **Create the virtual environment** (only once):
   ```
   python -m venv .venv
   ```
2. **Activate it**:
   - Windows (cmd/PowerShell): `.venv\Scripts\activate`
   - macOS/Linux: `source .venv/bin/activate`
3. **Install the package** in editable mode — this also installs the runtime dependencies (`mne`, `numpy`, `scipy`, `matplotlib`, `pandas`, `torch`, `scikit-learn`, `mne-lsl`) declared in `pyproject.toml`:
   ```
   pip install uv
   uv sync
   ```

Requires Python >= 3.12. The `.venv/` folder is git-ignored, so it never gets committed.

## Where to write code

- Put reusable Python modules under `src/` (e.g. `src/nova2026/data/`, `src/nova2026/architecture/`).
- Keep it importable: `from nova2026.data.mne_reader import ...`
- Put development / experimental scripts in `scripts/`, organized into subfolders (e.g. `scripts/dataproc/`, `scripts/training/`). Move code here into `src/nova2026/` once it stabilizes.

## Where to put datasets

- Use `datasets/<name>/` — the folder is already set up for this, and it's excluded from git so the large files stay local.
- Don't modify or reorganize the raw files; do preprocessing/cleaning in code instead.
- Save trained models / checkpoints under `models/` (also excluded from git).

## References (`refs.bib`)

All citations live in the BibTeX file `refs.bib` at the repo root. Whenever you read or use a source, add one entry per source and cite it in the docs via its key (e.g. `@hinss_2022_6874129`).

### Getting a BibTeX entry online

Most sources give you a ready-made BibTeX entry via a **Cite / Citation** button, select `BibTex` option. Then just paste the entry into `refs.bib`, keeping a unique, descriptive key.
