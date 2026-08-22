from collections.abc import Callable
from pathlib import Path

import mne

from nova2026.config import DATA_DIR


def load(set_file: str | Path) -> mne.io.BaseRaw:
    return mne.io.read_raw_eeglab(str(set_file), preload=True, verbose=False)


"""
The format of EEG recording files, the value list holds the
the required files for the given format.
"""
EEG_DATA_FORMAT = [".set", ".vhdr", ".dat", ".edf", ".bdf", ".cnt", ".txt", ".raw"]
# Unrecommanded practice:
# Do not name the constants with the same name as the
# method arguments.
# dataset_list = ["COG-BCI"]
# trial_filename_list = ["PVT.set"]
# resting_filename_list = ["RS_Beg_EC.set", "RS_Beg_EO.set"]


class Loader:
    """Recursively searches a dataset directory for EEG recording files.

    Args:
        dataset (str): Name of the dataset subfolder under the root directory.
        root (Path | None, optional): Base directory containing the dataset.
            Defaults to ``DATA_DIR``.

    Raises:
        FileNotFoundError: If the dataset directory does not exist.
    """

    def __init__(self, dataset: str, root: Path | None = None):
        self.root: Path = DATA_DIR if root is None else root
        self.dataset: Path = self.root / dataset
        self.query_cache: list[Path] | None = None
        if not (self.dataset).exists():
            raise FileNotFoundError(f"Dataset {self.dataset} not found in {self.root}")

    def search(
        self, query: str, query_type: str, query_path: Path | None = None
    ) -> list[Path]:
        """Recursively search for files matching a name or suffix query.

        Args:
            query (str): The filename stem (for ``query_type="name"``) or a
                file format like ``".set"`` (for ``query_type="suffix"``).
            query_type (str): Either ``"name"`` or ``"suffix"``.
            query_path (Path | None, optional): Directory to start the search
                from. Defaults to the dataset directory.

        Returns:
            list[Path]: Paths of all matching files.

        Raises:
            ValueError: If ``query_type`` is ``"suffix"`` and the query is not
                in :data:`EEG_DATA_FORMAT`.
        """
        query_path = query_path if query_path is not None else self.dataset

        # suffix check
        if query_type == "suffix" and not (query in EEG_DATA_FORMAT):
            raise ValueError(f"Unsupported file format: {query}")

        query_result: list[Path] = []
        for item_path in query_path.iterdir():
            # Recursive search
            if item_path.is_dir():
                query_result.extend(self.search(query, query_type, item_path))
                continue

            # check if the file matches the query
            hit_by_name = (query_type == "name") and (query in item_path.stem)
            hit_by_suffix = (query_type == "suffix") and (query in item_path.suffix)
            if hit_by_name or hit_by_suffix:
                query_result.append(item_path)

        self.query_cache = query_result.copy()
        return query_result

    def look_for(self, is_valid: Callable[[Path], bool]):
        """Filter the cached search results by a predicate.

        Args:
            is_valid (Callable[[Path], bool]): Predicate deciding whether a
                cached path is kept.

        Returns:
            list[Path]: The filtered subset of the cached results.

        Raises:
            ValueError: If no ``search`` has been performed yet.
        """
        if self.query_cache is None:
            raise ValueError("No query has been made yet")
        query_result = [p for p in self.query_cache if is_valid(p)]
        self.query_cache = query_result.copy()
        return query_result
