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


def test_forward_without_bias(fix_seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = "deterministic"

    inputs = torch.rand((1, 1, 3, 3)).to(device)
    model = BinarizedConv2d(in_channels=1,
                            out_channels=1,
                            kernel_size=3,
                            stride=1,
                            padding=0,
                            dilation=1,
                            groups=1,
                            bias=None,
                            padding_mode="zeros",
                            mode=mode).to(device)

    output = model(inputs)

    assert output.shape == (1, 1, 1, 1)


def test_forward_with_bias(fix_seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = "deterministic"

    inputs = torch.rand((1, 1, 3, 3)).to(device)
    model = BinarizedConv2d(in_channels=1,
                            out_channels=1,
                            kernel_size=3,
                            stride=1,
                            padding=0,
                            dilation=1,
                            groups=1,
                            bias=True,
                            padding_mode="zeros",
                            mode=mode).to(device)

    output = model(inputs)

    assert output.shape == (1, 1, 1, 1)


def test_forward_clipping(fix_seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = "deterministic"

    inputs = torch.rand((1, 1, 3, 3)).to(device)
    model = BinarizedConv2d(in_channels=1,
                            out_channels=1,
                            kernel_size=3,
                            stride=1,
                            padding=0,
                            dilation=1,
                            groups=1,
                            bias=True,
                            padding_mode="zeros",
                            mode=mode).to(device)
    with torch.no_grad():
        model.weight.mul_(100)

    output = model(inputs)

    with torch.no_grad():
        assert(model.weight.min() >= torch.tensor(-1.))
        assert(model.weight.max() >= torch.tensor(1.))
