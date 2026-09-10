"""Grouped training splits, null controls and duration-based controller metrics."""

import copy

import numpy as np


def check_split(training, validation, testing=()):
    """Reject reused trials or shared stimulus groups across partitions."""
    partitions = [training, validation, testing]
    seen_trials = set()
    seen_groups = set()
    for partition in partitions:
        trials = set()
        groups = set()
        for trial in partition:
            trials.add((trial.subject, trial.trial_id))
            groups.add(trial.group)
        if trials & seen_trials or groups & seen_groups:
            raise ValueError("Split reuses a trial or stimulus group.")
        seen_trials.update(trials)
        seen_groups.update(groups)


def inject_fault(trial, kind, start=8.0, duration=0.25):
    """Return a copy; original recordings are never changed."""
    result = copy.deepcopy(trial)
    selected = (result.timestamps >= start) & (result.timestamps < start + duration)
    if kind == "dropout":
        result.eeg[selected] = np.nan
    elif kind == "artifact":
        result.eeg[selected] += 2000
    elif kind == "mismatched":
        result.audio = np.roll(result.audio, round(7 * result.audio_rate), axis=0)
    elif kind != "none":
        raise ValueError("Unknown fault type.")
    return result


def selection_metrics(times, choices, labels):
    """Durations use each sample's following interval; unknown truth is excluded."""
    times = np.asarray(times)
    choices = np.asarray(choices)
    labels = np.asarray(labels)
    duration = np.diff(times, append=times[-1])
    known = np.isin(labels, [0, 1])
    neutral = choices == -1
    correct = known & (choices == labels)
    wrong = known & ~neutral & (choices != labels)
    total = float(duration[known].sum())
    changes = (choices[1:] != choices[:-1]) & (labels[1:] == labels[:-1])
    changes &= known[1:] & known[:-1]
    switch_delays = []
    missed = 0
    switches = []
    previous_label = None
    for index, label in enumerate(labels):
        if label not in (0, 1):
            continue
        if previous_label is not None and label != previous_label:
            switches.append(index)
        previous_label = label
    for position, index in enumerate(switches):
        end = switches[position + 1] if position + 1 < len(switches) else len(times)
        matches = np.flatnonzero(choices[index:end] == labels[index])
        if len(matches):
            switch_delays.append(float(times[index + matches[0]] - times[index]))
        else:
            missed += 1
    return {
        "known_seconds": total,
        "correct_emphasis_seconds": float(duration[correct].sum()),
        "wrong_suppression_seconds": float(duration[wrong].sum()),
        "neutral_seconds": float(duration[known & neutral].sum()),
        "coverage": float(duration[known & ~neutral].sum() / total) if total else None,
        "false_selection_changes": int(changes.sum()),
        "reported_switch_delays_seconds": switch_delays,
        "missed_switches": missed,
    }
