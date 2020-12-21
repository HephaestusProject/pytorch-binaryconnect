from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_lightning.core import LightningModule
from pytorch_lightning.metrics.functional import accuracy
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, StepLR

from src.utils import load_class


class TrainingContainer(LightningModule):
    def __init__(self, model: nn.Module, config: DictConfig):
        """Model Container for Training

        Args:
            model (nn.Module): model for train
            config (DictConfig): configuration with Omegaconf.DictConfig format for dataset/model/runner
        """
        super().__init__()
        self.model = model
        self.save_hyperparameters(config)
        self.config = config.runner
        print(self.config)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        opt_args = dict(self.config.optimizer.params)
        opt_args.update({"params": self.model.parameters()})
        opt = load_class(module=optim, name=self.config.optimizer.type, args=opt_args)

        scheduler_args = dict(self.config.scheduler.params)
        scheduler_args.update({"optimizer": opt})
        scheduler = load_class(
            module=optim.lr_scheduler,
            name=self.config.scheduler.type,
            args=scheduler_args,
        )

        result = {"optimizer": opt, "lr_scheduler": scheduler}
        if self.config.scheduler.params == "ReduceLROnPlateau":
            result.update({"monitor": self.config.scheduler.monitor})

        return result

    def shared_step(self, x, y):
        y_hat = self(x)
        loss = self.model.loss(y_hat, y)

        pred = torch.argmax(y_hat, dim=1)
        acc = accuracy(pred, y)

        return y_hat, loss, acc

    def training_step(self, batch, batch_idx):
        x, y = batch
        _, loss, acc = self.shared_step(x, y)

        self.log(
            name="train/accuracy",
            value=acc,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=False,
        )

        self.log(
            name="train_accuracy",
            value=acc,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=False,
        )

        return {
            "loss": loss,
            "train/accuracy": acc,
            "train_loss": loss,
            "train_accuracy": acc,
        }

    def training_epoch_end(self, training_step_outputs):
        acc, loss = 0, 0
        num_of_outputs = len(training_step_outputs)

        for log_dict in training_step_outputs:
            acc += log_dict["train/accuracy"]
            loss += log_dict["loss"]

        acc /= num_of_outputs
        loss /= num_of_outputs

        self.log(
            name="train/loss",
            value=loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            name="train_loss",
            value=loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            name="train/accuracy",
            value=acc,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            name="train_accuracy",
            value=acc,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

    def validation_step(self, batch, batch_idx):
        x, y = batch
        _, loss, acc = self.shared_step(x, y)

        return {
            "valid/loss": loss,
            "valid/accuracy": acc,
            "valid_loss": loss,
            "valid_accuracy": acc,
        }

    def validation_epoch_end(self, validation_step_outputs):
        acc, loss = 0, 0
        num_of_outputs = len(validation_step_outputs)

        for log_dict in validation_step_outputs:
            acc += log_dict["valid/accuracy"]
            loss += log_dict["valid/loss"]

        acc = acc / num_of_outputs
        loss = loss / num_of_outputs

        self.log(
            name="valid/loss",
            value=loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            name="valid/accuracy",
            value=acc,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        self.log(
            name="valid_loss",
            value=loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            name="valid_accuracy",
            value=acc,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
