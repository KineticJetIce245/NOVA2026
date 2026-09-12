"""Post-stimulus visual detection on the COG-BCI PVT recordings.

Predicts, from the occipital EEG in the 100-300 ms after stimulus onset,
whether the subject registered the stimulus. Complements the pre-stimulus
lapse-prediction arm in ``scripts/dataproc/``.

Modules
-------
``device``
    Channel sets, output paths and the checkpoint data-type tag.
``build_dataset``
    Builds the epoched checkpoint (stimulus and silence epochs, both label
    schemes) from the raw PVT recordings.
``models``
    ``EEGNetWindow``, an EEGNet whose temporal kernel and pooling are sized for
    a window of a few tens of samples.
``train_occipital``
    Leave-one-subject-out training and evaluation with the repository's
    ``SupervisedTrainer``.
"""
