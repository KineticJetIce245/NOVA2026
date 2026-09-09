"""Task-, model- and modality-agnostic supervised training loop.

The trainer owns the *procedure* (epoch loop, device placement, best-epoch
selection, test-time prediction) and lets the caller own the *task* through
injection points:

``predict_func``
    Maps raw model outputs to the predictions a metric consumes.
``Metric``
    Scores predictions and knows whether higher or lower is better.
``regularizer``
    Optional post-update projection applied to the model every batch.
``unpack``
    Splits a batch into ``(model_inputs, targets)`` when the default layout
    does not fit.
``collect``
    Extracts per-sample metadata (subject id, trial index, ...) from a batch so
    a :meth:`~nova2026.training.metrics.Metric.with_extras` metric can group
    predictions without depending on loader order.

Nothing here assumes a modality, an architecture or a task. Three contracts
must be honoured by the caller:

Batch contract
--------------
A batch is whatever the ``DataLoader`` yields. It is split into model inputs
and targets by :meth:`SupervisedTrainer.unpack` (overridable via ``uses``):

* ``Tensor``                       -> ``(tensor, None)``
* ``(inputs, targets)``            -> ``(inputs, targets)``
* ``(in1, in2, ..., targets)``     -> ``((in1, in2, ...), targets)``
* ``Mapping``                      -> inputs are the mapping minus the target
  entry, whose key is one of ``target``/``y``/``label``/``labels``/``targets``.

Inputs are dispatched as ``model(inputs)`` for a single object,
``model(*inputs)`` for a sequence and ``model(**inputs)`` for a mapping, so
single-input, multi-input and named-input models all work. Tensors nested
anywhere in a batch are moved to :attr:`device` automatically.

Split contract
--------------
``organize()`` must return a mapping. ``"train"`` is required; ``"val"`` and
``"test"`` are optional (omit the key or pass ``None``). Without ``"val"`` no
best-epoch selection happens and the final weights are used for testing.

Extras contract
---------------
If the metric declares ``needs_extras`` (build it with
:meth:`~nova2026.training.metrics.Metric.with_extras`), ``uses`` must also be
given a ``collect`` callable. Per-batch ``collect`` results are concatenated in
loader order and handed to the metric as a third argument, so per-subject or
per-group scores never have to reconstruct the ordering themselves. The test
split's extras are returned as ``"test_extras"``.

Call flow
---------
::

    fetch -> load_chkpt -> self.chkpt
    organize -> builder(self.chkpt) -> self.data {train, val?, test?}
    uses -> bind model/optimizer/criterion/regularizer/predict_func/metric/
            unpack/collect

    train
     +-- _ensure / _ensure_data                    # precondition guards
     +-- for each epoch
     |    +-- _train_once(train_loader)            # training path
     |    |     unpack -> _to_device -> _call_model -> criterion
     |    |     -> backward/step -> regularizer -> _num_samples (loss weight)
     |    +-- _predict(val_loader, predict_func[, collect])  # selection path
     |    |     unpack -> _to_device -> _call_model -> permutate
     |    |     -> _concat_extras -> metric(...) -> metric.is_better(...)
     |    |     -> _snapshot()
     |    +-- callbacks["epoch"](...)              # external observation
     +-- model.load_state_dict(best_state)         # roll back to the best epoch
     +-- _predict(test_loader,
     |            callbacks.get("test_output", predict_func)[, collect])
     +-- return {history, best_metric, best_epoch, test_preds, test_targets,
                 test_extras, best_state, model}
"""

import pickle
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from nova2026.training.metrics import Metric

__all__ = ["SupervisedTrainer", "load_chkpt"]

Batch = Any

# Keys recognised as the target entry of a mapping batch, in no particular
# order: exactly one of them must be present.
_TARGET_KEYS = ("target", "y", "label", "labels", "targets")


def _to_device(obj: Any, device: torch.device) -> Any:
    """Move every tensor nested in ``obj`` to ``device``, preserving structure."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, Mapping):
        return {key: _to_device(value, device) for key, value in obj.items()}
    if isinstance(obj, tuple):
        if hasattr(obj, "_fields"):  # namedtuple: keep the type
            return type(obj)(*(_to_device(value, device) for value in obj))
        return tuple(_to_device(value, device) for value in obj)
    if isinstance(obj, list):
        return [_to_device(value, device) for value in obj]
    return obj


def _num_samples(obj: Any) -> int | None:
    """Best-effort batch size of a nested structure (leading dim of a tensor)."""
    if torch.is_tensor(obj):
        return int(obj.shape[0]) if obj.dim() > 0 else None
    if isinstance(obj, Mapping):
        for value in obj.values():
            found = _num_samples(value)
            if found is not None:
                return found
    if isinstance(obj, (list, tuple)):
        for value in obj:
            found = _num_samples(value)
            if found is not None:
                return found
    return None


def _call_model(model: nn.Module, inputs: Any) -> Any:
    """Dispatch a single / positional / keyword model input."""
    if isinstance(inputs, Mapping):
        return model(**inputs)
    if isinstance(inputs, (list, tuple)):
        return model(*inputs)
    return model(inputs)


def _default_unpack(batch: Batch) -> tuple[Any, Any]:
    """Split a batch into ``(model_inputs, targets)`` (see module docstring)."""
    if isinstance(batch, Mapping):
        target_keys = [key for key in batch if key in _TARGET_KEYS]
        if len(target_keys) != 1:
            raise ValueError(
                "Mapping batches must contain exactly one target key "
                f"(one of {_TARGET_KEYS}); found {target_keys}. "
                "Pass unpack= to uses() for a custom layout."
            )
        target_key = target_keys[0]
        inputs = {key: value for key, value in batch.items() if key != target_key}
        return inputs, batch[target_key]
    if isinstance(batch, (list, tuple)):
        if len(batch) == 0:
            raise ValueError("Cannot unpack an empty batch.")
        if len(batch) == 1:
            return batch[0], None
        if len(batch) == 2:
            return batch[0], batch[1]
        # multi-input: everything but the last element is fed to the model.
        return tuple(batch[:-1]), batch[-1]
    return batch, None


def _argmax_predict(outputs: Any) -> Any:
    return torch.argmax(outputs, dim=1)


def _macro_f1_metric() -> Metric:
    return Metric(lambda yt, yp: f1_score(yt, yp, average="macro"), name="macro_f1")


def _concat_extras(extras: list[Any]) -> Any:
    """Align per-batch extras into one sequence ordered like the predictions.

    Tensors are concatenated, arrays concatenated, lists/tuples flattened.
    Anything else is left as the raw per-batch list, since it cannot be
    assumed to be indexable by sample.
    """
    if not extras:
        return extras
    if all(torch.is_tensor(item) for item in extras):
        return torch.cat(
            [item.reshape(-1) if item.dim() == 0 else item for item in extras], dim=0
        )
    if all(isinstance(item, np.ndarray) for item in extras):
        return np.concatenate([np.atleast_1d(item) for item in extras])
    if all(isinstance(item, (list, tuple)) for item in extras):
        return [value for item in extras for value in item]
    return extras


def load_chkpt(
    path: str | Path,
    *,
    map_location: Any = "cpu",
    weights_only: bool | None = None,
) -> dict:
    """Load a checkpoint, restricted unpickler first.

    ``weights_only=None`` (the default) tries ``weights_only=True`` and only
    falls back to arbitrary-object loading -- with a warning -- when the file
    really holds non-tensor Python objects (this project's dataset dumps do).
    Pass ``True`` to forbid the fallback or ``False`` to skip the attempt.

    ``map_location`` defaults to ``"cpu"`` so a checkpoint saved on a GPU can
    be inspected on a machine without one.
    """
    path = Path(path)
    if weights_only is None:
        try:
            return torch.load(path, map_location=map_location, weights_only=True)
        except pickle.UnpicklingError as exc:
            warnings.warn(
                f"{path.name} contains non-tensor objects; loading it executes "
                "arbitrary pickle code. Only do this for trusted checkpoints. "
                f"({exc})",
                RuntimeWarning,
                stacklevel=2,
            )
            return torch.load(path, map_location=map_location, weights_only=False)
    return torch.load(path, map_location=map_location, weights_only=weights_only)


class SupervisedTrainer:
    """Epoch loop with device placement, best-epoch selection and testing."""

    def __init__(self, device: str | torch.device | None = None, seed: int | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if seed is not None:
            torch.manual_seed(seed)
        self.model: nn.Module | None = None
        self.optimizer: Optimizer | None = None
        self.criterion: nn.Module | None = None
        self.regularizer: Callable | None = None
        self.predict_func: Callable | None = None
        self.metric: Metric | None = None
        self.data: Mapping[str, DataLoader | None] | None = None
        self.chkpt: dict | None = None
        # Batch -> (model_inputs, targets); replace via uses(unpack=...).
        # Always reset in uses(), never inherited from a previous fold.
        self.unpack: Callable[[Batch], tuple[Any, Any]] = _default_unpack
        # Batch -> per-sample metadata, gathered when the metric needs extras.
        self.collect: Callable[[Batch], Any] | None = None

    def fetch(
        self,
        path: str | Path,
        *,
        map_location: Any = "cpu",
        weights_only: bool | None = None,
    ) -> dict:
        """Load a dataset checkpoint into :attr:`chkpt` and return it."""
        self.chkpt = load_chkpt(
            path, map_location=map_location, weights_only=weights_only
        )
        return self.chkpt

    def organize(self, builder: Callable[[dict], Mapping[str, DataLoader | None]]) -> dict:
        """Build the split mapping (``train`` required, ``val``/``test`` optional)."""
        if self.chkpt is None:
            raise ValueError("No checkpoint loaded. Call fetch() before organize().")
        data = builder(self.chkpt)
        if not isinstance(data, Mapping):
            raise TypeError(
                "organize() builder must return a mapping of split name -> DataLoader."
            )
        self.data = data
        return dict(data)

    def uses(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        regularizer: Callable | None = None,
        *,
        predict_func: Callable | None = None,
        metric: Metric | None = None,
        unpack: Callable[[Batch], tuple[Any, Any]] | None = None,
        collect: Callable[[Batch], Any] | None = None,
    ) -> None:
        """Register the model, optimisation and task-specific hooks.

        ``unpack`` and ``collect`` are reset on every call, so a custom layout
        registered for one fold never leaks into the next. ``collect`` is
        required when ``metric`` declares ``needs_extras`` (see
        :meth:`nova2026.training.metrics.Metric.with_extras`).
        """
        if model is None:
            raise ValueError("Model must be provided")
        if optimizer is None:
            raise ValueError("Optimizer must be provided")
        if criterion is None:
            raise ValueError("Criterion must be provided")

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.regularizer = regularizer

        if predict_func is None:
            warnings.warn(
                "predict_func not provided; defaulting to argmax(dim=1), which "
                "is only correct for single-label classification.",
                RuntimeWarning,
                stacklevel=2,
            )
            predict_func = _argmax_predict
        self.predict_func = predict_func

        if metric is None:
            warnings.warn(
                "metric not provided; defaulting to macro F1, which is only "
                "correct for single-label classification.",
                RuntimeWarning,
                stacklevel=2,
            )
            metric = _macro_f1_metric()
        self.metric = metric

        if metric.needs_extras and collect is None:
            raise ValueError(
                f"Metric {metric.name!r} needs per-sample extras; pass collect= "
                "to uses() so they can be gathered from each batch."
            )

        self.unpack = unpack if unpack is not None else _default_unpack
        self.collect = collect

    def _ensure(self) -> None:
        missing = [
            name
            for name, value in (
                ("model", self.model),
                ("optimizer", self.optimizer),
                ("criterion", self.criterion),
                ("predict_func", self.predict_func),
                ("metric", self.metric),
            )
            if value is None
        ]
        if missing:
            raise ValueError(f"Call uses() first; missing: {', '.join(missing)}")

    def _ensure_data(self) -> None:
        if self.data is None:
            raise ValueError("Data must be provided. Call organize() first.")

    def _snapshot(self) -> dict:
        """Detached copy of the current weights (cheaper than deepcopy)."""
        return {key: value.detach().clone() for key, value in self.model.state_dict().items()}

    def _train_once(self, loader: DataLoader) -> float:
        self._ensure()
        self.model.train()
        reduction = getattr(self.criterion, "reduction", "mean")
        total, count = 0.0, 0
        # training
        for batch in loader:
            inputs, targets = self.unpack(batch)
            inputs = _to_device(inputs, self.device)
            targets = _to_device(targets, self.device)

            # backprop
            self.optimizer.zero_grad()
            outputs = _call_model(self.model, inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            # regularization
            if self.regularizer is not None:
                with torch.no_grad():
                    self.regularizer(self.model)

            # sample-weighted mean, correct for both "mean" and "sum" reductions
            size = _num_samples(targets) or _num_samples(inputs) or 1
            total += loss.item() if reduction == "sum" else loss.item() * size
            count += size

        if count == 0:
            raise ValueError("The training loader yielded no samples.")
        return total / count

    def _predict(
        self,
        loader: DataLoader,
        permutate: Callable | None = None,
        collect: Callable[[Batch], Any] | None = None,
    ):
        """Run inference over ``loader``.

        Returns ``(predictions, targets)``, or ``(predictions, targets, extras)``
        when ``collect`` is given. ``targets`` is ``None`` for unlabelled data.
        ``collect`` receives each raw batch and lets a metric see per-sample
        metadata (subject id, index, ...) without relying on loader order; the
        per-batch results are aligned with the predictions by
        :func:`_concat_extras`.
        """
        self._ensure()
        self.model.eval()
        raw, targets, extras = [], [], []
        # run predictions
        with torch.no_grad():
            for batch in loader:
                inputs, target = self.unpack(batch)
                inputs = _to_device(inputs, self.device)
                outputs = _call_model(self.model, inputs)
                if permutate is not None:
                    outputs = permutate(outputs)
                raw.append(outputs.detach().cpu())
                if target is not None:
                    targets.append(_to_device(target, torch.device("cpu")))
                if collect is not None:
                    extras.append(collect(batch))

        if not raw:
            raise ValueError("The prediction loader yielded no samples.")

        predictions = torch.cat(raw, dim=0).cpu().numpy()
        target_array = (
            torch.cat(targets, dim=0).cpu().numpy() if targets else None
        )
        if collect is not None:
            return predictions, target_array, _concat_extras(extras)
        return predictions, target_array

    def train(
        self,
        epochs: int,
        callbacks: Mapping[str, Callable] | None = None,
    ) -> dict:
        """Train for ``epochs`` epochs and test with the selected weights.

        ``callbacks`` is an optional mapping. ``"epoch"`` is called after every
        epoch with ``{epoch, train_loss, best_metric, best_epoch, best_state}``;
        ``"test_output"`` overrides the prediction post-processing used on the
        test split (defaults to :attr:`predict_func`).

        When the registered metric declares ``needs_extras``, the validation
        pass gathers ``collect`` from every batch and hands the aligned extras
        to the metric. ``collect`` is also applied to the test split, whose
        extras come back under ``"test_extras"``.
        """
        self._ensure()
        self._ensure_data()
        if epochs <= 0:
            raise ValueError("epochs must be positive.")

        callbacks = dict(callbacks or {})
        train_loader = self.data.get("train")
        if train_loader is None:
            raise ValueError(
                "data['train'] is required; got keys: "
                + ", ".join(map(str, self.data))
            )
        val_loader = self.data.get("val")
        test_loader = self.data.get("test")

        history = []
        best_metric, best_epoch, best_state = None, None, None

        for epoch in range(epochs):
            # train once
            train_loss = self._train_once(train_loader)
            log = {"epoch": epoch, "train_loss": train_loss}

            # perform evaluation if needed
            if val_loader is not None:
                if self.metric.needs_extras:
                    val_preds, val_targets, val_extras = self._predict(
                        val_loader, self.predict_func, collect=self.collect
                    )
                    val_metric = self.metric(val_targets, val_preds, val_extras)
                else:
                    val_preds, val_targets = self._predict(
                        val_loader, self.predict_func
                    )
                    val_metric = self.metric(val_targets, val_preds)
                log["metric"] = val_metric
                if self.metric.is_better(val_metric, best_metric):
                    best_metric = val_metric
                    best_state = self._snapshot()
                    best_epoch = epoch

            history.append(log)

            # perform callbacks for each epoch
            on_epoch = callbacks.get("epoch")
            if on_epoch is not None:
                on_epoch(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "best_metric": best_metric,
                        "best_epoch": best_epoch,
                        # a copy, so callbacks cannot corrupt the selection
                        "best_state": (
                            {key: value.clone() for key, value in best_state.items()}
                            if best_state is not None
                            else None
                        ),
                    }
                )

        # Testing
        state_to_use = best_state if best_state is not None else self.model.state_dict()
        self.model.load_state_dict(state_to_use)
        test_preds, test_targets, test_extras = None, None, None
        if test_loader is not None:
            test_permutate = callbacks.get("test_output", self.predict_func)
            if self.collect is not None:
                test_preds, test_targets, test_extras = self._predict(
                    test_loader, test_permutate, collect=self.collect
                )
            else:
                test_preds, test_targets = self._predict(test_loader, test_permutate)
        return {
            "history": history,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "test_preds": test_preds,
            "test_targets": test_targets,
            "test_extras": test_extras,
            "best_state": best_state,
            "model": self.model,
        }
