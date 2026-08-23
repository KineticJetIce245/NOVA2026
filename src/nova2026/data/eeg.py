from collections.abc import Callable
from pathlib import Path

import mne

from nova2026.config import DATA_DIR


def load(set_file: str | Path) -> mne.io.BaseRaw:
    """Load an EEGLAB ``.set`` file into a raw MNE object.

    Args:
        set_file (str | Path): Path to the ``.set`` file to load.

    Returns:
        mne.io.BaseRaw: The loaded raw data.
    """
    return mne.io.read_raw_eeglab(str(set_file), preload=True, verbose=False)


"""Supported formats of EEG recording files.

The list holds the file suffixes (e.g. ``".set"``) that are recognized
when searching with ``query_type="suffix"``.
"""
EEG_DATA_FORMAT = [".set", ".vhdr", ".dat", ".edf", ".bdf", ".cnt", ".txt", ".raw"]

LOAD_OPTS = {
    "eeglab": mne.io.read_raw_eeglab,
    "brainvision": mne.io.read_raw_brainvision,
    "edf": mne.io.read_raw_edf,
    "bdf": mne.io.read_raw_bdf,
    "cnt": mne.io.read_raw_cnt,
}


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

        def __str__(self) -> str:
            return f"[{self.tags}: {self.path}]"

        def __repr__(self) -> str:
            return self.__str__()

    class TaggedData:
        """Represents a tagged raw data object.

        Attributes:
            raw (mne.io.BaseRaw | None): The raw data object.
            tags (list[str] | None): Optional tags associated with the data.
        """

        def __init__(self, raw: mne.io.BaseRaw, tags: list[str]) -> None:
            self.raw: mne.io.BaseRaw = raw
            self.tags: list[str] = tags

        def is_tagged_with(self, tag: str) -> bool:
            return tag in self.tags

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
            list[DataFileRef]: References to all matching files.

        Raises:
            ValueError: If ``query_type`` is ``"suffix"`` and the query is not
                in :data:`EEG_DATA_FORMAT`.
        """
        query_path = query_path if query_path is not None else self.dataset

        # suffix check
        if query_type == "suffix" and not (query in EEG_DATA_FORMAT):
            raise ValueError(f"Unsupported file format: {query}")

        for item_path in query_path.iterdir():
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
            validator (Callable[[DataFileRef], bool]): Predicate deciding
                whether a cached reference is kept.

        Returns:
            list[DataFileRef]: The filtered subset of the cached results.
        """
        self.query_cache = [p for p in self.query_cache if validator(p)]
        return self.query_cache

    def tag(self, tagger: Callable[[DataFileRef], list[str]]):
        """Tag the cached search results.

        Args:
            tagger (Callable[[DataFileRef], list[str]]): Function that takes
                a data file reference and returns a list of tags.

        Returns:
            list[DataFileRef]: The tagged subset of the cached results.
        """
        for q in self.query_cache:
            q.tags = tagger(q)
        return self.query_cache

    def load(
        self,
        mode: str | None = None,
        loader: Callable[[DataFileRef], TaggedData] | None = None,
    ) -> list[TaggedData]:
        """Load the cached search results.

        Args:
            mode (str | None, optional): Name of a predefined loader in
                :data:`LOAD_OPTS` (e.g. ``"eeglab"``). Mutually exclusive
                with ``loader``.
            loader (Callable[[DataFileRef], TaggedData] | None, optional):
                Custom function that takes a DataFileRef object and returns
                a TaggedData object. Defaults to ``None``.

        Returns:
            list[TaggedData]: The loaded subset of the cached results.

        Raises:
            ValueError: If neither ``mode`` nor ``loader`` is given, or if
                ``mode`` is not a key of :data:`LOAD_OPTS`.
        """
        loader_func = None
        if loader is None:
            if mode is None:
                raise ValueError("Either mode or loader must be specified")
            else:
                mne_func = LOAD_OPTS.get(mode)
                if mne_func is None:
                    raise ValueError(f"Unknown mode: {mode}")
                loader_func = lambda drf: Loader.TaggedData(
                    raw=mne_func(drf.path, verbose=False), tags=drf.tags
                )
        else:
            loader_func = loader

        if loader_func is None:
            raise ValueError("Unable to obtain loader.")

        loaded_list: list[Loader.TaggedData] = []
        for q in self.query_cache:
            loaded_list.append(loader_func(q))
        return loaded_list
