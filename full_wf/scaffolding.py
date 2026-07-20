from abc import ABC
from dataclasses import dataclass

import copy
import torch
from torch import nn


def load_experiment_track(filepath, map_location=None):
    """Load experiment data from a safe dict checkpoint.

    New checkpoints are plain dictionaries. If a legacy checkpoint is loaded,
    this function converts it into the same dictionary shape when possible.
    """
    payload = torch.load(filepath, map_location=map_location, weights_only=False)

    if isinstance(payload, dict) and payload.get("format") == "experiment_track_v2":
        return payload

    # Best-effort conversion for legacy objects saved as CallbackContext.
    if hasattr(payload, "epoch") and hasattr(payload, "train_loss") and hasattr(payload, "val_loss"):
        model_state = None
        model_obj = getattr(payload, "model", None)
        if model_obj is not None and hasattr(model_obj, "state_dict"):
            model_state = model_obj.state_dict()

        return {
            "format": "experiment_track_legacy_converted",
            "epoch": int(payload.epoch),
            "train_loss": float(payload.train_loss),
            "val_loss": float(payload.val_loss),
            "stop_training": bool(getattr(payload, "stop_training", False)),
            "best_state": getattr(payload, "best_state", None),
            "model_state_dict": model_state,
            "extra": {},
        }

    return payload


# ==========================================================
# Callback Context
# ==========================================================
# Small shared object passed to callbacks instead of the full Trainer.

@dataclass
class CallbackContext:
    model: nn.Module
    epoch: int
    train_loss: float
    val_loss: float
    stop_training: bool = False
    best_state: dict | None = None


# ==========================================================
# Base Callback (no-op hooks)
# ==========================================================

class Callback(ABC):
    def on_train_begin(self, ctx):        pass
    def on_epoch_begin(self, ctx):        pass
    def on_epoch_end(self, ctx):          pass
    def on_train_end(self, ctx):          pass


# ==========================================================
# Concrete Callbacks
# ==========================================================
class Preliminary_information(Callback):

    def __init__(self, Xtrain_shape, ytrain_shape, Xval_shape, yval_shape):
        self.Xtrain_shape = Xtrain_shape
        self.ytrain_shape = ytrain_shape
        self.Xval_shape = Xval_shape
        self.yval_shape = yval_shape

    def on_train_begin(self, ctx):
        Xtrain_shape = self.Xtrain_shape
        ytrain_shape = self.ytrain_shape
        Xval_shape = self.Xval_shape
        yval_shape = self.yval_shape
        import torchinfo
        print("Training started.")
        print(f"Initial model state: {ctx.model.state_dict()}")
        print(f"Overall context: {ctx}")
        print(f"Xtrain shape: {Xtrain_shape}")
        print(f"ytrain shape: {ytrain_shape}")
        print(f"Xval shape: {Xval_shape}")
        print(f"yval shape: {yval_shape}")
        print("Model summary:")
        print(torchinfo.summary(ctx.model, input_data=torch.randn(*Xtrain_shape)))


class ProgressPrinter(Callback):

    def __init__(self, n=1):
        self.n = n

    def on_epoch_end(self, ctx):
        # only print the progress if the epoch is a multiple of n
        if ctx.epoch % self.n == 0:
            print(
                f"Epoch {ctx.epoch:4d} | "
                f"Train Loss = {ctx.train_loss:.4f} | "
                f"Val Loss = {ctx.val_loss:.4f}"
            )


class LossHistory(Callback):

    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def on_epoch_end(self, ctx):
        self.train_losses.append(ctx.train_loss)
        self.val_losses.append(ctx.val_loss)


class EarlyStopping(Callback):

    def __init__(
        self,
        patience=20,
        min_delta=0.0,
        mode="min",  # "min" for loss, "max" for metrics
        restore_best_weights=True,
    ):
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.counter = 0

    def _improved(self, value):
        if self.mode == "min":
            return value < (self.best_score - self.min_delta)
        return value > (self.best_score + self.min_delta)

    def on_epoch_end(self, ctx):
        current = ctx.val_loss

        if self._improved(current):
            self.best_score = current
            self.counter = 0
            if self.restore_best_weights:
                ctx.best_state = copy.deepcopy(ctx.model.state_dict())
        else:
            self.counter += 1

        if self.counter >= self.patience:
            ctx.stop_training = True


class ModelCheckpoint(Callback):

    def __init__(self, filepath, monitor="val_loss", mode="min"):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.best_score = float("inf") if mode == "min" else float("-inf")

    def _improved(self, value):
        if self.mode == "min":
            return value < self.best_score
        return value > self.best_score

    def on_epoch_end(self, ctx):
        current = getattr(ctx, self.monitor)
        if self._improved(current):
            self.best_score = current
            torch.save(ctx.model.state_dict(), self.filepath)

class ExperimentTrack(Callback):
    """Save experiment context as a plain dictionary checkpoint."""

    def __init__(self, filepath):
        self.filepath = filepath
    
    def on_train_end(self, ctx, **kwargs):
        payload = {
            "format": "experiment_track_v2",
            "epoch": int(ctx.epoch),
            "train_loss": float(ctx.train_loss),
            "val_loss": float(ctx.val_loss),
            "stop_training": bool(ctx.stop_training),
            "best_state": ctx.best_state,
            "model_state_dict": ctx.model.state_dict(),
            "extra": kwargs,
        }
        torch.save(payload, self.filepath)
    



# ==========================================================
# Trainer
# ==========================================================

class Trainer:

    def __init__(self, model, optimizer, criterion, callbacks=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.callbacks = list(callbacks) if callbacks is not None else []

    def fit(self, Xtrain, ytrain, Xval, yval, epochs ):
        ctx = CallbackContext(model=self.model, epoch=0, train_loss=float("nan"), val_loss=float("nan"))

        for cb in self.callbacks:
            cb.on_train_begin(ctx)

        for epoch in range(epochs):
            ctx.epoch = epoch

            for cb in self.callbacks:
                cb.on_epoch_begin(ctx)

            self.model.train()
            self.optimizer.zero_grad()
            y_pred = self.model(Xtrain)
            train_loss = self.criterion(y_pred, ytrain)
            train_loss.backward()
            self.optimizer.step()
            ctx.train_loss = float(train_loss.item())

            self.model.eval()
            with torch.no_grad():
                y_val_pred = self.model(Xval)
                val_loss = self.criterion(y_val_pred, yval)
                ctx.val_loss = float(val_loss.item())

            for cb in self.callbacks:
                cb.on_epoch_end(ctx)

            if ctx.stop_training:
                print(f"Early stopping at epoch {epoch}")
                break

        if ctx.best_state is not None:
            self.model.load_state_dict(ctx.best_state)

        for cb in self.callbacks:
            cb.on_train_end(ctx)

        return ctx
