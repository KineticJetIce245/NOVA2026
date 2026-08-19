from nova2026.data import eeg
from pathlib import Path
import numpy as np
import torch
import mne
import pandas as pd

from nova2026.config import(
    SAMPLE_RATE,
    SAMPLE_LENGTH,
    WINDOW_LENGTH,
    STEP_SIZE
)

from pre_processing_pipelines import pipelines

EEG_CHANNELS = [
    "Fp1",
    "Fz",
    "F3",
    "F7",
    "FT9",
    "FC5",
    "FC1",
    "C3",
    "T7",
    "CP5",
    "CP1",
    "Pz",
    "P3",
    "P7",
    "O1",
    "Oz",
    "O2",
    "P4",
    "P8",
    "TP10",
    "CP6",
    "CP2",
    "FCz",
    "C4",
    "T8",
    "FT10",
    "FC6",
    "FC2",
    "F4",
    "F8",
    "Fp2",
    "AF7",
    "AF3",
    "AFz",
    "F1",
    "F5",
    "FT7",
    "FC3",
    "C1",
    "C5",
    "TP7",
    "CP3",
    "P1",
    "P5",
    "PO7",
    "PO3",
    "POz",
    "PO4",
    "PO8",
    "P6",
    "P2",
    "CPz",
    "CP4",
    "TP8",
    "C6",
    "C2",
    "FC4",
    "FT8",
    "F6",
    "AF8",
    "AF4",
    "F2",
]

ROOT = Path("datasets/COG-BCI")
#SAMPLE_RATE = 128  # Hz
#SAMPLE_LENGTH = 2400  # ms
#WINDOW_LENGTH = 2000  # ms
#STEP_SIZE = 300  # ms


TRANSLATION = {
    "10": "PVT Start",
    "11": "PVT Trial/ISI Start",
    "12": "PVT ISI Error",
    "13": "PVT Stimulus",
    "14": "PVT Response",
    "15": "PVT  End",
    "boundary": "Boundary",
}


def load_subjects():
    subs = []
    for sub in ROOT.iterdir():
        if "sub" in sub.name and sub.is_dir():
            subs.append(sub.name)
    return np.array(subs)


def load_sessions(sub):
    session_name = ROOT / sub
    print(session_name)
    session_paths = []
    for s in session_name.iterdir():
        if s.is_dir():
            print(s.name)
            session_paths.append(s.name)
    return np.array(session_paths)
    #return np.array([s.name for s in (ROOT / sub).iterdir() if s.is_dir()])


def load_runs(sub, ses):
    print(ROOT / sub / ses)
    return eeg.load(ROOT / sub / ses / "eeg/PVT.set")


def get_trials(raw):
    trials = []
    stmls_tmstp = 0
    rsp_tmstp = 0
    err_tmstp = 0
    for ann in raw.annotations:
        # Evil fix for the onset
        tmstp = np.int32(ann["onset"] * 1000)
        if ann["description"] == "13":  # stimulus
            stmls_tmstp = tmstp
        elif ann["description"] == "14":  # response
            trials.append(
                np.array(
                    [
                        (tmstp - stmls_tmstp),
                        min(stmls_tmstp - rsp_tmstp, stmls_tmstp - err_tmstp),
                        stmls_tmstp,
                        tmstp,
                    ]
                )
            )
            rsp_tmstp = tmstp
        elif ann["description"] == "12":  # error
            trials.append(np.array([-1, -1, -1, tmstp]))
            err_tmstp = tmstp
    return np.array(trials)


def load_eeg_trials(data_list, label_list, meta_list, raw, trials, sub, ses, label):
    for trial in trials:
        stmls_time = trial[2]
        meta_list.append((sub, ses, trial[0]))
        label_list.append(label)

        batch = []
        for i in range(1):  # 1 windows
            tmax = (stmls_time - i * STEP_SIZE - 100) / 1000  # starting from -0.1s
            idxmax = raw.time_as_index(tmax)[0]
            idxmin = idxmax - int(WINDOW_LENGTH / 1000 * SAMPLE_RATE)
            batch.append(
                raw.get_data(picks=EEG_CHANNELS, start=idxmin, stop=idxmax) * 1e6
            )
        data_list.append(np.stack(batch, axis=0))


def label_data():
    data_list = []
    label_list = []
    meta_list = []

    # Finds sub-# folder
    subs = load_subjects()
    for sub in subs:
        # Finds ses-S# folder
        sessions = load_sessions(sub)
        for ses in sessions:
            # Finds actual PVT.set dataset
            raw = load_runs(sub, ses)
            
            trials = get_trials(raw)
            raw.load_data()
            # Filter 0.5Hz - 45Hz
            pipelines.default_pipeline(raw)

            raw.pick(EEG_CHANNELS)

            qualified_mask = trials[:, 1] > SAMPLE_LENGTH + 200  # 200ms as buffer

            cleaned_trials = trials[qualified_mask]
            cleaned_trials = cleaned_trials[
                np.argsort(cleaned_trials[:, 0])
            ]  # Sort by RT
            n = int(len(cleaned_trials) * 0.1)

            bottom_10_percent = cleaned_trials[-n:]
            other = cleaned_trials[:-n]

            load_eeg_trials(
                data_list, label_list, meta_list, raw, bottom_10_percent, sub, ses, 1
            )

            load_eeg_trials(data_list, label_list, meta_list, raw, other, sub, ses, 0)
            

def check_annotations() -> None:
    subs = load_subjects()
    print(subs)
    for sub in subs:
        if sub == "sub-01":
            sessions = load_sessions(sub)
            for session in sessions:
                raw = load_runs(sub, session)
                print(raw.annotations.to_data_frame())
        else:
            return

check_annotations()

'''
    data_array = np.array(data_list)
    label_array = np.array(label_list)
    meta_array = np.array(meta_list)

    print(data_array.shape, label_array.shape, meta_array.shape)

    data_tensor = torch.from_numpy(data_array).float()
    label_tensor = torch.from_numpy(label_array).long()


    torch.save(
        {
            "data": data_tensor,
            "labels": label_tensor,
            "metadata": meta_array,
        },
        ROOT / "PVT_data_window_1.pt",
    )


label_data()
'''

# raw = load_runs("sub-01", "ses-S1")
# raw.plot(n_channels=5, scalings="auto", title="EEG 波形")
#
# input("waiting...")
