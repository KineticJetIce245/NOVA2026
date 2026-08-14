# NOVA2026
## Folder structure

```
NOVA2026/
├── datasets/     Raw datasets (git-ignored)
├── documents/    Learning material (PDFs) + typst sources
├── references/   Reference PDFs
├── src/          Python code
├── refs.bib      Bibliography (BibTeX)
└── README.md
```

| Folder | Purpose |
| --- | --- |
| `src/` | **Your code goes here.** Organize it into subpackages (e.g. `src/data/`, `src/models/`). See `src/data/mne_reader.py` for an example of reading an EEGLAB `.set` file with MNE. |
| `datasets/` | **Put raw datasets here.** The folder already exists and is the recommended location. Keep raw data intact (`.set`/`.fdt`, `.cnt`, behavioral logs, channel locations). |
| `documents/` | Handouts, challenge briefs, and learning notes (PDF), plus their `typst/` sources. |
| `references/` | Reference papers / device documentation (e.g. `attentivU.pdf`). |
| `refs.bib` | Bibliography — see below. |

> Note: `datasets/*` is git-ignored (only `.gitkeep` is tracked), so large raw data never gets committed.

## Where to write code

- Put all Python modules under `src/` (e.g. `src/data/`, `src/analysis/`).
- Keep it importable: `from src.data.mne_reader import ...`
- Start from `src/data/mne_reader.py` to see the existing pattern for loading data with MNE.

## Where to put datasets

- Use `datasets/<name>/` — the folder is already set up for this, and it's excluded from git so the large files stay local.
- Don't modify or reorganize the raw files; do preprocessing/cleaning in code instead.

## References (`refs.bib`)

All citations live in the BibTeX file `refs.bib` at the repo root. Whenever you read or use a source, add one entry per source and cite it in the docs via its key (e.g. `@hinss_2022_6874129`).

### Getting a BibTeX entry online

Most sources give you a ready-made BibTeX entry via a **Cite / Citation** button, select `BibTex` option. Then just paste the entry into `refs.bib`, keeping a unique, descriptive key.
