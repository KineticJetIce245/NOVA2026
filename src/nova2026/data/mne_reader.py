import mne
import pandas as pd
import numpy as np
from nova2026.config import DATASETS, DATASET_DIR

file_paths = DATASETS
# raw = mne.io.read_raw_eeglab(file_path, preload=True)

# print(raw.info)
# raw.plot(n_channels=30, duration=10, block=True)
# print(raw.annotations)

"""
for file in file_paths:
    print(DATASET_DIR / file)
    montage = mne.channels.read_custom_montage(DATASET_DIR / file)
    positions = montage.get_positions()
    print(positions["ch_pos"])
"""

list_of_data_frames = []

for file in file_paths:
    output = np.array([])
    f = DATASET_DIR / file
    with open(f, "r") as elc:
        started = False
        for line in elc:
            line = line.rstrip("\n")
            if "Positions" in line and "NumberPositions" not in line:
                started = True
                continue
            if line.strip() == "":
                continue
            if "Labels" in line:
                break
            if started:
                output = np.append(output, line.split())

    output = output.reshape(-1, 5)
    output = output[:, [0, 2, 3, 4]]
    df = pd.DataFrame(output, columns=["label", "x", "y", "z"])
    list_of_data_frames.append(df)
    print(df)

ch = input("check for channel: ")


def fnx_rows_identical(dfs):
    """
    Check if the row(s) with label == 'Fnx' are exactly the same in all DataFrames.
    Returns True if all are identical, False otherwise.
    """
    # Extract the row for 'Fnx' from each DataFrame (assuming unique)
    fnx_rows = []
    for i, df in enumerate(dfs):
        # Filter rows where label == 'Fnx'
        mask = df["label"] == ch
        if mask.sum() == 0:
            print(f"DataFrame {i} does not contain {ch}")
            continue
        elif mask.sum() > 1:
            print(f"DataFrame {i} contains multiple {ch} rows")
            continue
        # Get the row as a Series (excluding the label column, or including it?)
        row = df.loc[mask].iloc[0]  # full row with label
        fnx_rows.append(row)

    # Compare all rows to the first one
    print(fnx_rows)


# Usage
result = fnx_rows_identical(list_of_data_frames)
