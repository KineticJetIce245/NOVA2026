import copy
from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset


def load_chkpt(dataset: Path) -> dict:
    return torch.load(dataset, weights_only=False)


class SupervisedTrainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fetch(self, dataset: Path) -> None:
        self.chkpt = load_chkpt(dataset)

    def organize(self, organizor: Callable[[dict], dict]) -> dict:
        self.data = organizor(self.chkpt)
        return self.data

    def uses(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        regularizer: Callable,
    ) -> None:
        if model is None:
            raise ValueError("Model must be provided")
        self.model = model.to(self.device)
        if optimizer is None:
            raise ValueError("Optimizer must be provided")
        self.optimizer = optimizer
        if criterion is None:
            raise ValueError("Criterion must be provided")
        self.criterion = criterion
        self.regularizer = regularizer

    def _ensure(self):
        if self.model is None:
            raise ValueError("Model must be provided")
        if self.optimizer is None:
            raise ValueError("Optimizer must be provided")
        if self.criterion is None:
            raise ValueError("Criterion must be provided")

    def _ensure_data(self):
        if self.data is None:
            raise ValueError("Data must be provided")

    def _train_once(self, loader: DataLoader) -> float:
        self._ensure()
        self.model.train()
        total, count = 0.0, 0
        # training
        for batch_data, batch_target in loader:
            batch_data, batch_target = (
                batch_data.to(self.device),
                batch_target.to(self.device),
            )

            # backprop
            self.optimizer.zero_grad()
            outputs = self.model(batch_data)
            loss = self.criterion(outputs, batch_target)
            loss.backward()
            self.optimizer.step()

            # regularization
            if self.regularizer != None:
                with torch.no_grad():
                    self.regularizer(self.model)

            total += loss.item() * batch_data.size(0)
            count += batch_data.size(0)
        return total / count

    def _predict(self, loader: DataLoader, permutate: Callable):
        self._ensure()
        self.model.eval()
        raw, targets = [], []
        # run predictions
        with torch.no_grad():
            for batch_data, batch_target in loader:
                batch_data = batch_data.to(self.device)
                outputs = self.model(batch_data)
                if permutate is not None:
                    outputs = permutate(outputs)
                raw.extend(outputs.cpu().numpy())
                targets.extend(batch_target.cpu().numpy())
        return raw, targets

    def _evaluate_loss(self, loader: DataLoader) -> float:
        self._ensure()
        self.model.eval()
        total, count = 0.0, 0
        # run tests
        with torch.no_grad():
            for batch_data, batch_target in loader:
                batch_data, batch_target = (
                    batch_data.to(self.device),
                    batch_target.to(self.device),
                )
                loss = self.criterion(self.model(batch_data), batch_target)
                total += loss.item() * batch_data.size(0)
                count += batch_data.size(0)
        return total / count

    def train(
        self,
        epochs: int,
        permutates: dict[str, Callable],
    ) -> dict:
        try:
            train_loader = self.data["train"]
            val_loader = self.data["val"]
            test_loader = self.data["test"]
        except KeyError as e:
            raise ValueError(
                f"KeyError{e}: Data not properly organized, call organize() first."
            )

        history = []
        best_val_loss, best_epoch, best_state = float("inf"), -1, None

        for epoch in range(epochs):
            # train once
            train_loss = self._train_once(train_loader)
            log = {"epoch": epoch, "train_loss": train_loss}

            # perform evaluation if needed
            if val_loader is not None:
                val_loss = self._evaluate_loss(val_loader)
                log["val_loss"] = val_loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(self.model.state_dict())
                    best_epoch = epoch

            history.append(log)

            # perform callbacks for each epoch
            if permutates is None:
                raise ValueError("permutates must be provided")
            per_epoch_consumer = permutates.get("epoch", lambda x: None)
            per_epoch_consumer(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "best_so_far": best_val_loss,
                    "best_epoch": best_epoch,
                    "best_state": best_state,
                }
            )

        test_preds, test_targets = None, None
        state_to_use = best_state if best_state is not None else self.model.state_dict()
        self.model.load_state_dict(state_to_use)
        if test_loader is not None:
            test_preds, test_targets = self._predict(
                test_loader, permutates.get("test_output", lambda x: x)
            )

        return {
            "history": history,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "test_preds": test_preds,
            "test_targets": test_targets,
            "best_state": best_state,
            "model": self.model,
        }
