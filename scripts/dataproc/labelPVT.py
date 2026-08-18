from nova2026.data import eeg
from pathlib import Path
import numpy as np
import torch

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
SAMPLE_RATE = 128  # Hz
SAMPLE_LENGTH = 2400  # ms
WINDOW_LENGTH = 2000  # ms
STEP_SIZE = 100  # ms


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
    return np.array([s.name for s in (ROOT / sub).iterdir() if s.is_dir()])


def load_runs(sub, ses):
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


def load_eeg_trials(data_list, label_list, subject_list, raw, trials, sub, ses):
    for trial in trials:
        stmls_time = trial[2]
        subject_list.append((sub, ses))
        if trial[0] > 500:
            label_list.append(1)
        else:
            label_list.append(0)

        batch = []
        for i in range(4):
            tmax = (stmls_time - (i + 1) * STEP_SIZE) / 1000  # starting from -0.1s
            idxmax = raw.time_as_index(tmax)[0]
            idxmin = idxmax - int(WINDOW_LENGTH / 1000 * SAMPLE_RATE)
            batch.append(raw.get_data(picks=EEG_CHANNELS, start=idxmin, stop=idxmax))

        data_list.append(np.stack(batch, axis=0))


def label_data():
    data_list = []
    label_list = []
    subject_list = []

    cln_trials_num = 0
    pos_num = 0
    all_trials = []

    subs = load_subjects()
    for sub in subs:
        sessions = load_sessions(sub)
        for ses in sessions:
            raw = load_runs(sub, ses)
            trials = get_trials(raw)
            raw.load_data()
            # Filter 0.5Hz - 45Hz
            raw.filter(0.5, 45.0, fir_design="firwin", verbose=False)
            raw.resample(SAMPLE_RATE)

            raw.pick(EEG_CHANNELS)

            qualified_mask = trials[:, 1] > SAMPLE_LENGTH + 200  # 200ms as buffer
            cleaned_trials = trials[qualified_mask]
            load_eeg_trials(
                data_list, label_list, subject_list, raw, cleaned_trials, sub, ses
            )

            # Stats
            cln_trials_num += cleaned_trials.shape[0]
            all_trials.append((sub, ses, cleaned_trials))
            pos_num += np.sum(cleaned_trials[:, 0] > 500)

    data_array = np.array(data_list)
    label_array = np.array(label_list)
    subject_array = np.array(subject_list)

    print(data_array.shape, label_array.shape, subject_array.shape)

    data_tensor = torch.from_numpy(data_array).float()
    label_tensor = torch.from_numpy(label_array).long()

    torch.save(
        {
            "data": data_tensor,
            "labels": label_tensor,
            "subjects": subject_array,
        },
        ROOT / "PVT_data.pt",
    )


label_data()
