from nova2026.architecture.cnn import EEGNet
from nova2026.architecture.lossfun import FocalLoss
from nova2026.config import DATA_DIR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score
import torch.nn.functional as F
import torch
import copy
import numpy as np

ROOT = DATA_DIR / "COG-BCI"
DATASET = ROOT / "PVT_data_2000ms_200ms.pt"

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
K = 90

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def max_norm_(param, max_value=1.0, eps=1e-8):
    """Applies max-norm constraint to the parameter."""
    with torch.no_grad():
        norm = param.norm(2)
        if norm > max_value:
            param.mul_(max_value / (norm + eps))


def load_data(path=DATASET):
    checkpoint = torch.load(path, weights_only=False)
    data = checkpoint["data"]
    labels = checkpoint["labels"]
    meta = checkpoint["metadata"]

    subjects = meta[:, 0]
    rt = meta[:, 2].astype(float)
    return data, labels, subjects, rt


def train_one_fold(x_train, y_train, x_val, y_val):
    x_train = torch.as_tensor(x_train, dtype=torch.float32)
    y_train = torch.as_tensor(y_train, dtype=torch.long)
    x_val = torch.as_tensor(x_val, dtype=torch.float32)
    y_val = torch.as_tensor(y_val, dtype=torch.long)
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset = TensorDataset(x_val, y_val)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = EEGNet(chn=62).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = FocalLoss(gamma=3.0, alpha=[1, 3.5], reduction="mean")
    # criterion = torch.nn.CrossEntropyLoss()

    best_f1 = 0.0
    best_model_state = None
    best_epoch = -1

    print(f"Train samples: {len(x_train)}, Test samples: {len(x_val)}")

    for epoch in range(1, EPOCHS + 1):
        # ----- Training -----
        model.train()
        train_loss = 0.0
        for batch_data, batch_target in train_loader:
            batch_data, batch_target = batch_data.to(DEVICE), batch_target.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_target)
            loss.backward()
            optimizer.step()

            # max-norm
            with torch.no_grad():
                max_norm_(model.depthwise_conv.weight, max_value=1.0)
                max_norm_(model.classifier.weight, max_value=0.25)

            train_loss += loss.item() * batch_data.size(0)

        train_loss /= len(x_train)

        # ----- Testing -----
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch_data, batch_target in test_loader:
                batch_data, batch_target = (
                    batch_data.to(DEVICE),
                    batch_target.to(DEVICE),
                )
                outputs = model(batch_data)
                loss = criterion(outputs, batch_target)
                val_loss += loss.item() * batch_data.size(0)

                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(batch_target.cpu().numpy())

                # probs = F.softmax(outputs, dim=1)
                # print("Probs:", probs[:5])
                # print("Targets:", batch_target[:5])

        val_loss /= len(x_val)
        val_f1 = f1_score(all_targets, all_preds, average="macro")
        val_acc = accuracy_score(all_targets, all_preds)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1(macro): {val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

    print(f"Best F1: {best_f1:.4f} at epoch {best_epoch}")
    return best_f1


def main():
    # Data Loading
    data, labels, sub_names, rt = load_data(DATASET)
    n_trials = data.shape[0]
    print(f"Total trials: {n_trials}, Total subjects: {len(np.unique(sub_names))}")

    # Prepare LOSO
    unique_subs = np.unique(sub_names)
    fold_results = []

    for test_sub in unique_subs:
        print(f"\n=== Fold: test subject = {test_sub} ===")

        # Split data based on subject
        test_trial_idx = np.where(sub_names == test_sub)[0]
        train_trial_idx = np.where(sub_names != test_sub)[0]

        selected_train_trials = []
        train_subs = np.unique(sub_names[train_trial_idx])
        # For each subject in the training set
        for sub in train_subs:
            sub_idx = np.where(sub_names == sub)[0]
            sub_labels = labels[sub_idx].numpy()
            sub_rt = rt[sub_idx]

            pos_local_all = np.where(sub_labels == 1)[0]
            neg_local_all = np.where(sub_labels == 0)[0]

            # Cap BOTH classes at K — don't let one class flood the training set
            k_pos = min(K, len(pos_local_all))
            k_neg = min(K, len(neg_local_all))

            pos_local = pos_local_all[
                np.argsort(sub_rt[pos_local_all])[-k_pos:]
            ]  # slowest RT
            neg_local = neg_local_all[
                np.argsort(sub_rt[neg_local_all])[:k_neg]
            ]  # fastest RT

            selected_local = np.concatenate([pos_local, neg_local])
            selected_train_trials.extend(sub_idx[selected_local])

        selected_train_trials = np.array(selected_train_trials)

        print(
            "Train label distribution:",
            dict(
                zip(
                    *np.unique(
                        labels[selected_train_trials].numpy(), return_counts=True
                    )
                )
            ),
        )

        # Shape data and labels into windows for training and testing
        train_data_windows = data[selected_train_trials].reshape(-1, 62, 256)
        train_labels_windows = np.repeat(labels[selected_train_trials], 1)
        test_data_windows = data[test_trial_idx].reshape(-1, 62, 256)
        test_labels_windows = np.repeat(labels[test_trial_idx], 1)

        print(
            "Test label distribution:",
            dict(zip(*np.unique(test_labels_windows, return_counts=True))),
        )

        X_train = train_data_windows
        y_train = train_labels_windows
        X_test = test_data_windows
        y_test = test_labels_windows

        best_f1 = train_one_fold(X_train, y_train, X_test, y_test)
        fold_results.append(best_f1)

    # Outputs
    print("\n================ LOSO Results ================")
    for sub, f1 in zip(unique_subs, fold_results):
        print(f"Subject {sub}: Best F1 = {f1:.4f}")
    mean_f1 = np.mean(fold_results)
    std_f1 = np.std(fold_results)
    print(f"Mean F1: {mean_f1:.4f} ± {std_f1:.4f}")


if __name__ == "__main__":
    main()
