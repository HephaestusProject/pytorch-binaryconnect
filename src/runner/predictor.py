from typing import Tuple

import torch
import torch.nn as nn
import torchvision
from omegaconf import DictConfig
from PIL import Image

from src.model import net as Net
from src.utils import load_class


def build_model(model_conf: DictConfig):
    return load_class(module=Net, name=model_conf.type, args={"model_config": model_conf})


class Predictor(torch.nn.Module):
    def __init__(self, config: DictConfig) -> None:
        """Model Container for Training

        Args:
            model (nn.Module): model for train
            config (DictConfig): configuration with Omegaconf.DictConfig format for dataset/model/runner
        """
        super().__init__()
        print(f"=======CONFIG=======")
        print(config)
        print(f"====================")
        self.model: nn.Module = build_model(model_conf=config.model)

    def forward(self, x):
        return self.model.single_inference(x)

    def preprocess(self, image: Image):
        return torchvision.transforms.ToTensor()(image).unsqueeze(0)
