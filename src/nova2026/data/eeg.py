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

    class DataFileRef:
        """Represents a reference to a data file.

        Attributes:
            path (Path): The path to the data file.
            tags (list[str] | None): Optional tags associated with the data file.
        """

        def __init__(self, path: Path, tags: list[str] | None) -> None:
            self.path: Path = path
            self.tags: list[str] | None = tags

    class TaggedData:
        """Represents a tagged raw data object.

        Attributes:
            raw (mne.io.BaseRaw | None): The raw data object.
            tags (list[str] | None): Optional tags associated with the data.
        """

        def __init__(self) -> None:
            self.raw: mne.io.BaseRaw | None = None
            self.tags: list[str] | None = None

    def __init__(self, dataset: str, root: Path | None = None):
        """Initialize the Loader.

        Args:
            dataset (str): Name of the dataset subfolder under the root directory.
            root (Path | None, optional): Base directory containing the dataset.
                Defaults to ``DATA_DIR``.

        Raises:
            FileNotFoundError: If the dataset directory does not exist.
        """
        self.root: Path = DATA_DIR if root is None else root
        self.dataset: Path = self.root / dataset
        self.query_cache: list[Loader.DataFileRef] = []
        if not (self.dataset).exists():
            raise FileNotFoundError(f"Dataset {self.dataset} not found in {self.root}")

    def clear(self):
        """Clear the cached search results."""
        self.query_cache = []

    def search(
        self, query: str, query_type: str, query_path: Path | None = None
    ) -> list[DataFileRef]:
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

        for item_path in query_path.iterdir():
            print(item_path)
            # Recursive search
            if item_path.is_dir():
                self.search(query, query_type, item_path)
                continue

            # check if the file matches the query
            hit_by_name = (query_type == "name") and (query in item_path.stem)
            hit_by_suffix = (query_type == "suffix") and (query in item_path.suffix)
            if hit_by_name or hit_by_suffix:
                self.query_cache.append(Loader.DataFileRef(item_path, None))

        return self.query_cache

    def refine(self, validator: Callable[[DataFileRef], bool]):
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
        self.query_cache = [p for p in self.query_cache if validator(p)]
        return self.query_cache

    def tag(self, tagger: Callable[[Path], list[str]]):
        """Tag the cached search results.

        Args:
            tagger (Callable[[Path], list[str]]): Function that takes a path
                and returns a list of tags.

        Returns:
            list[Path]: The tagged subset of the cached results.
        """
        if self.query_cache is None:
            raise ValueError("No query has been made yet")

        for q in self.query_cache:
            q.tags = tagger(q.path)
        return self.query_cache

    def load(self, loader: Callable[[DataFileRef], TaggedData]):
        """Load the cached search results.

        Args:
            loader (Callable[[DataFileRef], TaggedData]): Function that takes a
                DataFileRef object and returns a TaggedData object.

        Returns:
            list[TaggedData]: The loaded subset of the cached results.
        """
        loaded_list: list[Loader.TaggedData] = []
        for q in self.query_cache:
            loader_list = loader(q)
            loaded_list.append(loader_list)
        return loaded_list
