from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_lightning.core import LightningModule
from pytorch_lightning.metrics.functional import accuracy
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR, StepLR, ReduceLROnPlateau

from src.utils import load_class


class Runner(LightningModule):
    def __init__(self, model: nn.Module, config: DictConfig):
        super().__init__()
        self.model = model
        self.save_hyperparameters(config)
        self.config = config.runner
        print(self.hparams)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        opt_args = dict(self.config.optimizer.params)
        opt_args.update({"params": self.model.parameters()})

        opt = load_class(module=optim, name=self.config.optimizer.type, args=opt_args)

        scheduler_args = dict(self.config.scheduler.params)
        scheduler_args.update({"optimizer": opt})
        scheduler = load_class(
            module=optim.lr_scheduler, name=self.config.scheduler.type, args=scheduler_args
        )

        result = {"optimizer": opt, "lr_scheduler": scheduler}
        if self.config.scheduler.params == "ReduceLROnPlateau":
            result.update({"monitor": self.config.scheduler.monitor})

        return result

    def _comm_step(self, x, y):
        y_hat = self(x)
        loss = self.model.loss(y_hat, y)

        pred = torch.argmax(y_hat, dim=1)
        acc = accuracy(pred, y)

        return y_hat, loss, acc

    def training_step(self, batch, batch_idx):
        x, y = batch
        _, loss, acc = self._comm_step(x, y)

        self.log(
            name="train_acc", value=acc, on_step=True, on_epoch=False, prog_bar=True, logger=False
        )

        return {"loss": loss, "train_acc": acc}

    def training_epoch_end(self, training_step_outputs):
        acc, loss = 0, 0
        num_of_outputs = len(training_step_outputs)

        for log_dict in training_step_outputs:
            acc += log_dict["train_acc"]
            loss += log_dict["loss"]

        acc /= num_of_outputs
        loss /= num_of_outputs

        self.log(name="loss", value=loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log(
            name="train_acc", value=acc, on_step=False, on_epoch=True, prog_bar=False, logger=True
        )

    def validation_step(self, batch, batch_idx):
        x, y = batch
        _, loss, acc = self._comm_step(x, y)

        return {"val_loss": loss, "val_acc": acc}

    def validation_epoch_end(self, validation_step_outputs):
        acc, loss = 0, 0
        num_of_outputs = len(validation_step_outputs)

        for log_dict in validation_step_outputs:
            acc += log_dict["val_acc"]
            loss += log_dict["val_loss"]

        acc = acc / num_of_outputs
        loss = loss / num_of_outputs

        self.log(
            name="val_loss", value=loss, on_step=False, on_epoch=True, prog_bar=False, logger=True
        )
        self.log(
            name="val_acc", value=acc, on_step=False, on_epoch=True, prog_bar=False, logger=True
        )

    def test_step(self, batch, batch_idx):
        result = self.validation_step(batch, batch_idx)
        result.rename_keys({"val_loss": "loss", "val_acc": "acc"})

        return result
