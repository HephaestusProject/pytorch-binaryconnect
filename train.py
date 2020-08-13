"""
Usage:
    main.py train [options]
    main.py train (-h | --help)
Options:
    --dataset-config <dataset config path>  Path to YAML file for dataset configuration  [default: conf/dataset/dataset.yml] [type: path]
    --model-config <model config path>  Path to YAML file for model configuration  [default: conf/model/model.yml] [type: path]
    --runner-config <runner config path>  Path to YAML file for model configuration  [default: conf/runner/runner.yml] [type: path]            
    -h --help  Show this.
"""
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torchvision.transforms as transforms
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (Callback, EarlyStopping,
                                         LearningRateLogger, ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from src.dataset.dataset_registry import DataSetRegistry
from src.model.net import DeepHash
from src.runner.runner import Runner
from src.utils import get_next_version


def get_config(hparams: Dict) -> DictConfig:
    config = OmegaConf.create()

    config_dir = Path(".")
    model_config = OmegaConf.load(config_dir / hparams.get("--model-config"))
    dataset_config = OmegaConf.load(config_dir / hparams.get("--dataset-config"))
    runner_config = OmegaConf.load(config_dir / hparams.get("--runner-config"))

    config.update(model=model_config, dataset=dataset_config, runner=runner_config)
    OmegaConf.set_readonly(config, True)

    return config

def get_checkpoint_callback(config: DictConfig) -> Union[Callback, List[Callback]]:
    root_dir = Path(config.runner.experiments.output_dir) / Path(config.runner.experiments.name)
    next_version = get_next_version(root_dir)
    checkpoint_dir = root_dir.joinpath(next_version)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_prefix = f"{config.model.type}"
    checkpoint_suffix = "_{epoch:02d}-{tr_loss:.2f}-{val_loss:.2f}-{tr_acc:.2f}-{val_acc:.2f}"

    checkpoint_path = checkpoint_dir.joinpath(checkpoint_prefix + checkpoint_suffix)
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path, save_top_k=2, save_weights_only=True
    )

    return checkpoint_callback


def get_logger_and_callbacks(
    config: DictConfig,
) -> Tuple[WandbLogger, Union[Callback, List[Callback]]]:

    root_dir = Path(config.runner.experiments.output_dir) / Path(config.runner.experiments.name)
    next_version = get_next_version(root_dir)
    checkpoint_dir = root_dir.joinpath(next_version)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    wandb_logger = WandbLogger(
        name=str(config.runner.experiments.name),
        save_dir=str(checkpoint_dir),
        offline=True,
        version=next_version,
    )

    checkpoint_prefix = f"{config.model.type}"
    checkpoint_suffix = "_{epoch:02d}-{tr_loss:.2f}-{val_loss:.2f}-{tr_acc:.2f}-{val_acc:.2f}"
    checkpoint_path = checkpoint_dir.joinpath(checkpoint_prefix + checkpoint_suffix)
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path, save_top_k=2, save_weights_only=True
    )

    return wandb_logger, checkpoint_callback

def get_data_loaders(config: DictConfig) -> Tuple[DataLoader, DataLoader]:

    dataset = DataSetRegistry.get(config.dataset.type)
    train_dataset = dataset(
        root=config.dataset.params.path.train,
        train=True,
        transform=transforms.ToTensor(),
        download=True,
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.runner.dataloader.params.batch_size,
        num_workers=config.runner.dataloader.params.num_workers,
        drop_last=True,
        shuffle=True,
    )
    test_dataset = dataset(
        root=config.dataset.params.path.test,
        train=False,
        transform=transforms.ToTensor(),
        download=True,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=config.runner.dataloader.params.batch_size,
        num_workers=config.runner.dataloader.params.num_workers,
        drop_last=False,
        shuffle=False,
    )
    return train_dataloader, test_dataloader

def train(hparams: dict):
    config = get_config(hparams=hparams)

    wandb_logger, checkpoint_callback = get_logger_and_callbacks(config=config)
    lr_logger = LearningRateLogger()

    early_stop_callback = EarlyStopping(
        monitor="val_acc", min_delta=0.00, patience=10, verbose=True, mode="max"
    )

    train_dataloader, test_dataloader = get_data_loaders(config=config)

    model = DeepHash(hash_bits=config.model.params.hash_bits)
    runner = Runner(model=model, config=config)

    trainer = Trainer(
        distributed_backend=config.runner.trainer.distributed_backend,
        fast_dev_run=False,
        gpus=config.runner.trainer.gpus,
        amp_level="O2",
        logger=wandb_logger,
        row_log_interval=10,
        callbacks=[lr_logger],
        early_stop_callback=early_stop_callback,
        checkpoint_callback=checkpoint_callback,
        max_epochs=config.runner.max_epochs,
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
    trainer.fit(
        model=runner, train_dataloader=train_dataloader, val_dataloaders=test_dataloader,
    )
