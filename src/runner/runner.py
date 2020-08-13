import torch
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_lightning.core import LightningModule
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR


class Runner(LightningModule):
    def __init__(self, model: nn.Module, config: DictConfig):
        super().__init__()
        self.model = model
        self.hparams.update({"dataset": f"{config.dataset.type}"})
        self.hparams.update({"model": f"{config.model.type}"})
        self.hparams.update(config.model.params)
        self.hparams.update(config.runner.dataloader.params)
        self.hparams.update({"optimizer": f"{config.runner.optimizer.params.type}"})
        self.hparams.update(config.runner.optimizer.params)
        self.hparams.update({"scheduler": f"{config.runner.scheduler.type}"})
        self.hparams.update({"scheduler_gamma": f"{config.runner.scheduler.params.gamma}"})
        self.hparams.update(config.runner.trainer.params)
        print(self.hparams)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        opt = SGD(params=self.model.parameters(), lr=self.hparams.learning_rate)
        scheduler = MultiStepLR(opt, milestones=[self.hparams.max_epochs], gamma=self.hparams.scheduler_gamma)
        return [opt], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = cross_entropy(y_hat, y)

        prediction = torch.argmax(y_hat, dim=1)
        acc = (y == prediction).float().mean()

        return {"loss": loss, "train_acc": acc}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["train_acc"] for x in outputs]).mean()
        tqdm_dict = {"train_acc": avg_acc, "train_loss": avg_loss}
        return {**tqdm_dict, "progress_bar": tqdm_dict}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = cross_entropy(y_hat, y)
        prediction = torch.argmax(y_hat, dim=1)
        number_of_correct_pred = torch.sum(y == prediction).item()
        return {"val_loss": loss, "n_correct_pred": number_of_correct_pred, "n_pred": len(x)}

    def validation_epoch_end(self, outputs):
        total_count = sum([x["n_pred"] for x in outputs])
        total_n_correct_pred = sum([x["n_correct_pred"] for x in outputs])
        total_loss = torch.stack([x["val_loss"] * x["n_pred"] for x in outputs]).sum()
        val_loss = total_loss / total_count
        val_acc = total_n_correct_pred / total_count
        tqdm_dict = {"val_acc": val_acc, "val_loss": val_loss}
        return {**tqdm_dict, "progress_bar": tqdm_dict}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = cross_entropy(y_hat, y)
        prediction = torch.argmax(y_hat, dim=1)
        number_of_correct_pred = torch.sum(y == prediction).item()
        return {"loss": loss, "n_correct_pred": number_of_correct_pred, "n_pred": len(x)}

    def test_epoch_end(self, outputs):
        total_count = sum([x["n_pred"] for x in outputs])
        total_n_correct_pred = sum([x["n_correct_pred"] for x in outputs])
        total_loss = torch.stack([x["loss"] * x["n_pred"] for x in outputs]).sum().item()
        test_loss = total_loss / total_count
        test_acc = total_n_correct_pred / total_count
        return {"loss": test_loss, "acc": test_acc}
