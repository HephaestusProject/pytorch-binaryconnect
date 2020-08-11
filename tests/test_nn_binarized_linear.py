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


forward_test_case = [
    # (device, test_input, test_bias, test_mode, exptected_shape)
    ("cpu", torch.rand((1, 10)), False, "deterministic", (1, 20)),
    (
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        torch.rand((1, 10)),
        False,
        "deterministic",
        (1, 20),
    ),
    ("cpu", torch.rand((1, 10)), True, "deterministic", (1, 20)),
    (
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        torch.rand((1, 10)),
        True,
        "deterministic",
        (1, 20),
    ),
    ("cpu", torch.rand((1, 10)), False, "stochastic", (1, 20)),
    (
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        torch.rand((1, 10)),
        False,
        "stochastic",
        (1, 20),
    ),
    ("cpu", torch.rand((1, 10)), True, "stochastic", (1, 20)),
    (
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        torch.rand((1, 10)),
        True,
        "stochastic",
        (1, 20),
    ),
]


@pytest.mark.parametrize(
    "device, test_input, test_bias, test_mode, exptected_shape", forward_test_case
)
def test_forward(fix_seed, device, test_input, test_bias, test_mode, exptected_shape):
    test_input = test_input.to(device)
    model = BinarizedLinear(10, 20, bias=test_bias, mode=test_mode).to(device)

    assert model(test_input).shape == exptected_shape


clipping_test_case = [
    (
        "cpu",
        torch.rand((1, 10)),
        False,
        "determistic",
        torch.tensor(1.0),
        torch.tensor(-1.0),
    ),
    (
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        torch.rand((1, 10)),
        False,
        "determistic",
        torch.tensor(1.0),
        torch.tensor(-1.0),
    ),
    (
        "cpu",
        torch.rand((1, 10)),
        True,
        "determistic",
        torch.tensor(1.0),
        torch.tensor(-1.0),
    ),
    (
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        torch.rand((1, 10)),
        True,
        "determistic",
        torch.tensor(1.0),
        torch.tensor(-1.0),
    ),
    (
        "cpu",
        torch.rand((1, 10)),
        False,
        "stochastic",
        torch.tensor(1.0),
        torch.tensor(-1.0),
    ),
    (
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        torch.rand((1, 10)),
        False,
        "stochastic",
        torch.tensor(1.0),
        torch.tensor(-1.0),
    ),
    (
        "cpu",
        torch.rand((1, 10)),
        True,
        "stochastic",
        torch.tensor(1.0),
        torch.tensor(-1.0),
    ),
    (
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        torch.rand((1, 10)),
        True,
        "stochastic",
        torch.tensor(1.0),
        torch.tensor(-1.0),
    ),
]


@pytest.mark.parametrize(
    "device, test_input, test_bias, test_mode, exptected_max_value, exptected_min_value",
    clipping_test_case,
)
def test_clipping(
    fix_seed,
    device,
    test_input,
    test_bias,
    test_mode,
    exptected_max_value,
    exptected_min_value,
):

    test_input = test_input.to(device)
    model = BinarizedLinear(10, 20, bias=test_bias, mode=test_mode).to(device)

    with torch.no_grad():
        model.weight.mul_(100)

    model(test_input)

    with torch.no_grad():
        assert model.weight.min() >= exptected_min_value
        assert model.weight.max() <= exptected_max_value
