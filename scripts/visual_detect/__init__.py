"""Post-stimulus visual detection on the COG-BCI PVT recordings.

Predicts, from the posterior EEG after stimulus onset, whether the subject
registered the stimulus. Standalone: it shares no code or preprocessing with the
pre-stimulus attention pipeline in ``scripts/dataproc/``.

Modules
-------
``pipeline``
    The preprocessing pipeline (notch, band-pass, resample, optional
    AttentivU-style scaling) and its defaults for this task.
``device``
    Channel sets (8 / 17 / 62), output paths and the checkpoint data-type tag.
``build_dataset``
    Builds the epoched checkpoint (stimulus and silence epochs, both label
    schemes) from the raw PVT recordings.
``decode``
    The algorithmic arm: template matching, xDAWN, log-Euclidean Riemannian
    tangent space, and two linear baselines. These keep the time axis.
``block_decode``
    Does averaging K consecutive trials rescue the decoder?
``metric_report``
    Scores one decoder several ways, to show why the ranking is AUC.
``incremental``
    Whether the EEG predicts the next block's reaction time beyond reaction
    time itself.
``models``
    ``EEGNetWindow``, an EEGNet whose temporal kernel and pooling are sized for
    a window of a few tens of samples. The comparison arm.
``train_occipital``
    Leave-one-subject-out and per-subject training for the network arm.
"""
