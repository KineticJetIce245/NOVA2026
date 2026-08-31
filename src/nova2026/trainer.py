from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset


def load_chkpt(dataset: Path) -> dict:
    return torch.load(dataset, weights_only=False)


class Trainer:
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
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        # criterion = FocalLoss(gamma=3.0, alpha=[1, 3.5], reduction="mean")
        # criterion = torch.nn.CrossEntropyLoss()
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

    def _train_once(self, train_loader):
        self._ensure()
        self.model.train()
        train_loss = 0.0
        for batch_data, batch_target in train_loader:
            batch_data, batch_target = (
                batch_data.to(self.device),
                batch_target.to(self.device),
            )

            self.optimizer.zero_grad()
            outputs = self.model(batch_data)
            loss = self.criterion(outputs, batch_target)
            loss.backward()
            self.optimizer.step()

            if self.regularizer != None:
                with torch.no_grad():
                    self.regularizer(self.model)

            train_loss += loss.item() * batch_data.size(0)
        return train_loss

    def _test_once(self, all_preds, all_targets, test_loader):
        self._ensure()
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data, batch_target in test_loader:
                batch_data, batch_target = (
                    batch_data.to(self.device),
                    batch_target.to(self.device),
                )
                outputs = self.model(batch_data)
                loss = self.criterion(outputs, batch_target)
                val_loss += loss.item() * batch_data.size(0)

                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(batch_target.cpu().numpy())
        return val_loss

    def train(self, batch_size: int, epochs: int):
        try:
            train_loader = DataLoader(
                TensorDataset(self.data["data"]["train"], self.data["labels"]["train"]),
                batch_size=batch_size,
                shuffle=True,
            )
            test_loader = DataLoader(
                TensorDataset(self.data["data"]["test"], self.data["labels"]["test"]),
                batch_size=batch_size,
                shuffle=False,
            )
        except KeyError:
            raise ValueError("Data must be organized before training.")

        for epoch in range(epochs):
            all_preds, all_targets = [], []
            self._train_once(train_loader)
            self._test_once(all_preds, all_targets, test_loader)
