from .pipeline import DefaultPipe, Pipeline

__all__ = ["DefaultPipe", "Loader", "Pipeline", "load_eeg", "save_chkpt"]


def __getattr__(name):
    """Keep the auditory path independent of the optional torch-based EEG loader."""
    if name in ("Loader", "load_eeg", "save_chkpt"):
        from importlib import import_module
        value = getattr(import_module('.eeg', __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(name)
