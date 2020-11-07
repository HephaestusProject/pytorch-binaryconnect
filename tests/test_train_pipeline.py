import os
import sys

import pytest
import pytorch_lightning
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateLogger

from src.nn.binarized_conv2d import BinarizedConv2d
from src.runner.runner import Runner
from src.utils import get_data_loaders
from train import build_model


@pytest.fixture(scope="module")
def fix_seed():
    pytorch_lightning.seed_everything(777)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tearup_mlp_config():
    return {
        "dataset": {
            "type": "MNIST",
            "params": {
                "root": "data/",
                "train": None,
                "transform": None,
                "target_transform": None,
                "download": True,
            },
        },
        "model": {
            "type": "BinaryLinear",
            "params": {
                "width": 28,
                "height": 28,
                "channels": 1,
                "in_feature": 784,
                "classes": 10,
                "mode": "stochastic",
                "feature_layers": {
                    "linear": [
                        {
                            "in_feature": 784,
                            "out_feature": 1024,
                            "bias": True,
                            "batch_norm": True,
                            "activation": {"type": "ReLU", "args": {}},
                            "mode": "stochastic",
                        },
                        {
                            "in_feature": 1024,
                            "out_feature": 1024,
                            "bias": True,
                            "batch_norm": True,
                            "activation": {"type": "ReLU", "args": {}},
                            "mode": "stochastic",
                        },
                        {
                            "in_feature": 1024,
                            "out_feature": 1024,
                            "bias": True,
                            "batch_norm": True,
                            "activation": None,
                            "mode": "stochastic",
                        },
                    ]
                },
                "output_layer": {
                    "type": "Linear",
                    "args": {"in_features": 1024, "out_features": 10, "bias": True},
                },
            },
        },
        "runner": {
            "type": "Runner",
            "dataloader": {"type": "DataLoader", "params": {"num_workers": 48, "batch_size": 200}},
            "optimizer": {"type": "SGD", "params": {"lr": 0.01, "momentum": 0}},
            "scheduler": {"type": "ExponentialLR", "params": {"gamma": 0.96, "verbose": True}},
            "trainer": {
                "type": "Trainer",
                "params": {
                    "max_epochs": 2,
                    "gpus": -1,
                    "distributed_backend": "dp",
                    "fast_dev_run": False,
                    "amp_level": "02",
                    "row_log_interval": 10,
                    "weights_summary": "top",
                    "reload_dataloaders_every_epoch": False,
                    "resume_from_checkpoint": None,
                    "benchmark": False,
                    "deterministic": True,
                    "num_sanity_val_steps": 5,
                    "overfit_batches": 0.0,
                    "precision": 32,
                    "profiler": True,
                },
            },
            "earlystopping": {
                "type": "EarlyStopping",
                "params": {"monitor": "val_acc", "mode": "max", "patience": 10, "verbose": True},
            },
            "experiments": {"name": "binaryconnect_mlp", "output_dir": "output/runs"},
        },
    }


def tearup_conv_config():
    return {
        "dataset": {
            "type": "CIFAR10",
            "params": {
                "root": "data/",
                "train": None,
                "transform": None,
                "target_transform": None,
                "download": True,
            },
        },
        "model": {
            "type": "BinaryConv",
            "params": {
                "width": 32,
                "height": 32,
                "channels": 3,
                "classes": 10,
                "mode": "stochastic",
                "feature_layers": {
                    "conv": [
                        {
                            "in_channels": 3,
                            "out_channels": 128,
                            "kernel_size": 3,
                            "stride": 1,
                            "padding": 0,
                            "bias": True,
                            "padding_mode": "zeros",
                            "batch_norm": True,
                            "activation": {"type": "ReLU", "args": {}},
                            "pool": None,
                            "mode": "stochastic",
                        },
                        {
                            "in_channels": 128,
                            "out_channels": 128,
                            "kernel_size": 3,
                            "stride": 1,
                            "padding": 0,
                            "bias": True,
                            "padding_mode": "zeros",
                            "batch_norm": True,
                            "activation": {"type": "ReLU", "args": {}},
                            "pool": {
                                "type": "MaxPool2d",
                                "args": {"kernel_size": [2, 2], "padding": 0},
                            },
                            "mode": "stochastic",
                        },
                        {
                            "in_channels": 128,
                            "out_channels": 256,
                            "kernel_size": 3,
                            "stride": 1,
                            "padding": 0,
                            "bias": True,
                            "padding_mode": "zeros",
                            "batch_norm": True,
                            "activation": {"type": "ReLU", "args": {}},
                            "pool": None,
                            "mode": "stochastic",
                        },
                        {
                            "in_channels": 256,
                            "out_channels": 256,
                            "kernel_size": 3,
                            "stride": 1,
                            "padding": 0,
                            "bias": True,
                            "padding_mode": "zeros",
                            "batch_norm": True,
                            "activation": {"type": "ReLU", "args": {}},
                            "pool": {
                                "type": "MaxPool2d",
                                "args": {"kernel_size": [2, 2], "padding": 0},
                            },
                            "mode": "stochastic",
                        },
                        {
                            "in_channels": 256,
                            "out_channels": 512,
                            "kernel_size": 3,
                            "stride": 1,
                            "padding": 0,
                            "bias": True,
                            "padding_mode": "zeros",
                            "batch_norm": True,
                            "activation": {"type": "ReLU", "args": {}},
                            "pool": None,
                            "mode": "stochastic",
                        },
                        {
                            "in_channels": 512,
                            "out_channels": 512,
                            "kernel_size": 3,
                            "stride": 1,
                            "padding": 0,
                            "bias": True,
                            "padding_mode": "zeros",
                            "batch_norm": True,
                            "activation": {"type": "ReLU", "args": {}},
                            "pool": None,
                            "mode": "stochastic",
                        },
                    ],
                    "linear": [
                        {
                            "in_feature": 512,
                            "out_feature": 1024,
                            "bias": True,
                            "batch_norm": True,
                            "activation": None,
                            "mode": "stochastic",
                        },
                        {
                            "in_feature": 1024,
                            "out_feature": 1024,
                            "bias": True,
                            "batch_norm": True,
                            "activation": {"type": "ReLU", "args": {}},
                            "mode": "stochastic",
                        },
                    ],
                },
                "output_layer": {
                    "type": "Linear",
                    "args": {"in_features": 1024, "out_features": 10, "bias": True},
                },
            },
        },
        "runner": {
            "type": "Runner",
            "dataloader": {"type": "DataLoader", "params": {"num_workers": 48, "batch_size": 50}},
            "optimizer": {"type": "Adam", "params": {"lr": 0.001}},
            "scheduler": {"type": "ExponentialLR", "params": {"gamma": 0.96, "verbose": True}},
            "trainer": {
                "type": "Trainer",
                "params": {
                    "max_epochs": 1000,
                    "gpus": -1,
                    "distributed_backend": "ddp",
                    "fast_dev_run": False,
                    "amp_level": "02",
                    "row_log_interval": 10,
                    "weights_summary": "top",
                    "reload_dataloaders_every_epoch": False,
                    "resume_from_checkpoint": None,
                    "benchmark": False,
                    "deterministic": True,
                    "num_sanity_val_steps": 5,
                    "overfit_batches": 0.0,
                    "precision": 32,
                    "profiler": True,
                },
            },
            "earlystopping": {
                "type": "EarlyStopping",
                "params": {"mode": "max", "patience": 10, "verbose": True},
            },
            "experiments": {"name": "binaryconnect_conv", "output_dir": "output/runs"},
        },
    }


train_test_case = [
    # (config, gpus)
    (tearup_mlp_config(), None),
    (tearup_mlp_config(), 0),
    (tearup_conv_config(), None),
    (tearup_conv_config(), 0),
]


# LightningModule의 end step(train end, valid end, test end)을 잘못짠듯
@pytest.mark.parametrize("config, gpus", train_test_case)
def test_train_pipeline(fix_seed, config, gpus):
    config = OmegaConf.create(config)

    train_dataloader, test_dataloader = get_data_loaders(config=config)
    lr_logger = LearningRateLogger()
    model = build_model(model_conf=config.model)
    runner = Runner(model=model, config=config.runner)

    trainer = Trainer(
        distributed_backend=config.runner.trainer.distributed_backend,
        fast_dev_run=True,
        gpus=gpus,
        amp_level="O2",
        row_log_interval=10,
        callbacks=[lr_logger],
        max_epochs=1,
        weights_summary="top",
        reload_dataloaders_every_epoch=False,
        resume_from_checkpoint=None,
        benchmark=False,
        deterministic=True,
        num_sanity_val_steps=5,
        overfit_batches=0.0,
        precision=32,
        profiler=True,
    )

    trainer.fit(model=runner, train_dataloader=train_dataloader, val_dataloaders=test_dataloader)
