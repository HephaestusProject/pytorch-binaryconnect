import os
import sys

import pytest
import pytorch_lightning
import torch

from src.nn.binarized_conv2d import BinarizedConv2d


@pytest.fixture(scope="module")
def fix_seed():
    pytorch_lightning.seed_everything(777)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
