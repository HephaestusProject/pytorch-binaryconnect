"""
Usage:
    main.py train [options]
    main.py train (-h | --help)
Options:
    --dataset-config <dataset config path>  Path to YAML file for dataset configuration  [default: conf/mlp/dataset/dataset.yml] [type: path]
    --model-config <model config path>  Path to YAML file for model configuration  [default: conf/mlp/model/model.yml] [type: path]
    --runner-config <runner config path>  Path to YAML file for model configuration  [default: conf/mlp/runner/runner.yml] [type: path]            
    -h --help  Show this.
"""
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torchvision.transforms as transforms
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, EarlyStopping, LearningRateLogger, ModelCheckpoint

from torch.utils.data import DataLoader
from src.model.net import BinaryLinear, BinaryConv
from src.runner.runner import Runner
from src.utils import (
    get_next_version,
    get_config,
    get_log_dir,
    get_checkpoint_callback,
    get_wandb_logger,
    get_early_stopper,
    get_data_loaders,
    load_class,
)
from src.model import net as Net


def build_model(model_conf: DictConfig):
    return load_class(module=Net, name=model_conf.type, args={"model_config": model_conf})


def train(hparams: dict):
    config_list = ["--dataset-config", "--model-config", "--runner-config"]
    config = get_config(hparams=hparams, options=config_list)

    log_dir = get_log_dir(config=config)
    log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = get_checkpoint_callback(log_dir=log_dir, config=config)
    wandb_logger = get_wandb_logger(log_dir=log_dir, config=config)
    lr_logger = LearningRateLogger()
    early_stop_callback = get_early_stopper(
        early_stopping_config=config.runner.earlystopping.params
    )

    train_dataloader, test_dataloader = get_data_loaders(config=config)

    model = build_model(model_conf=config.model)
    exit()
    """
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
    """
