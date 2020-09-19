from typing import Tuple, Any, Dict, List, Union
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateLogger,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_next_version(root_dir: Path):
    version_prefix = "v"
    if not root_dir.exists():
        next_version = 0
    else:
        existing_versions = []
        for child_path in root_dir.iterdir():
            if child_path.is_dir() and child_path.name.startswith(version_prefix):
                existing_versions.append(int(child_path.name[len(version_prefix) :]))

        if len(existing_versions) == 0:
            last_version = -1
        else:
            last_version = max(existing_versions)

        next_version = last_version + 1
    return f"{version_prefix}{next_version:0>3}"


def get_config(hparams: Dict, options: List) -> DictConfig:

    config: DictConfig = OmegaConf.create()

    for option in options:
        option_config: DictConfig = OmegaConf.load(hparams.get(option))
        config.update(option_config)

    OmegaConf.set_readonly(config, True)

    return config


def get_log_dir(config: DictConfig) -> Path:
    root_dir = Path(config.runner.experiments.output_dir) / Path(config.runner.experiments.name)
    next_version = get_next_version(root_dir)
    run_dir = root_dir.joinpath(next_version)

    return run_dir


def get_checkpoint_callback(log_dir: Path, config: DictConfig) -> Union[Callback, List[Callback]]:
    checkpoint_prefix = f"{config.model.type}"
    checkpoint_suffix = "_{epoch:02d}-{tr_loss:.2f}-{val_loss:.2f}-{tr_acc:.2f}-{val_acc:.2f}"

    checkpoint_path = log_dir.joinpath(checkpoint_prefix + checkpoint_suffix)
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path, save_top_k=2, save_weights_only=True
    )

    return checkpoint_callback


def get_wandb_logger(log_dir: Path, config: DictConfig) -> Tuple[WandbLogger]:
    next_version = str(log_dir.parts[-1])
    ids = log_dir.parts[-1]
    wandb_logger = WandbLogger(
        id=ids,
        name=str(config.runner.experiments.name),
        save_dir=str(log_dir),
        offline=False,
        version=next_version,
        project=str(config.runner.experiments.name),
    )

    return wandb_logger


def get_early_stopper(early_stopping_config: DictConfig):
    return EarlyStopping(
        monitor=early_stopping_config.monitor,
        min_delta=0.00,
        patience=early_stopping_config.patience,
        verbose=early_stopping_config.verbose,
        mode=early_stopping_config.mode,
    )


def get_data_loaders(config: DictConfig) -> Tuple[DataLoader, DataLoader]:

    args = dict(config.dataset.params)
    args["train"] = True
    args["transform"] = transforms.ToTensor()
    train_dataset = load_class(module=torchvision.datasets, name=config.dataset.type, args=args)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.runner.dataloader.params.batch_size,
        num_workers=config.runner.dataloader.params.num_workers,
        drop_last=True,
        shuffle=True,
    )

    args["train"] = False

    test_dataset = load_class(module=torchvision.datasets, name=config.dataset.type, args=args)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=config.runner.dataloader.params.batch_size,
        num_workers=config.runner.dataloader.params.num_workers,
        drop_last=False,
        shuffle=False,
    )
    return train_dataloader, test_dataloader


def load_class(module: Any, name: str, args: Dict):
    return getattr(module, name)(**args)
