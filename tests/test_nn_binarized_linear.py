import os
import sys

import pytest
import pytorch_lightning
import torch

from src.nn.binarized_linear import BinarizedLinear


@pytest.fixture(scope="module")
def fix_seed():
    pytorch_lightning.seed_everything(777)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test_forward_without_bias(fix_seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = "deterministic"

    inputs = torch.rand((1, 10)).to(device)
    model = BinarizedLinear(10, 20, bias=False, mode=mode).to(device)

    output = model(inputs)

    assert output.shape == (1, 20)


def test_forward_with_bias(fix_seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = "deterministic"

    inputs = torch.rand((1, 10)).to(device)
    model = BinarizedLinear(10, 20, bias=True, mode=mode).to(device)

    output = model(inputs)

    assert output.shape == (1, 20)


def test_forward_clipping(fix_seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = "deterministic"

    inputs = torch.rand((1, 10)).to(device)
    model = BinarizedLinear(10, 20, bias=False, mode=mode).to(device)

    with torch.no_grad():
        model.weight.mul_(100)

    model(inputs)

    with torch.no_grad():
        assert model.weight.min() >= torch.tensor(-1.0)
        assert model.weight.max() <= torch.tensor(1.0)
