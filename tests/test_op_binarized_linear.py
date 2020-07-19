import os
import sys
import pytest

import torch
import pytorch_lightning

current_file_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
if current_file_path not in sys.path:
    sys.path.append(current_file_path)
    from binaryconnect.ops.binarized_linear import binary_linear


def test_deterministic_binarized_linear_non_bias():
    pytorch_lightning.seed_everything(777)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    inputs = torch.randn((3, 3))
    weights = torch.randn((3, 3))

    result = binary_linear(inputs,
                           weights,
                           None,
                           "deterministic")

    rights = torch.tensor([[0.6937,  2.6833,  1.8804],
                           [-1.0636, -0.2879,  0.8230],
                           [-1.3459, -1.2886,  1.5963]])

    assert torch.allclose(input=result,
                          other=rights,
                          rtol=1e-04,
                          atol=1e-04,
                          equal_nan=True)


def test_deterministic_binarized_linear_bias():
    pytorch_lightning.seed_everything(777)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    inputs = torch.randn((3, 3))
    weights = torch.randn((3, 3))

    result = binary_linear(inputs,
                           weights,
                           torch.tensor([1.]),
                           "deterministic")

    rights = torch.tensor([[1.6937,  3.6833,  2.8804],
                           [-0.0636,  0.7121,  1.8230],
                           [-0.3459, -0.2886,  2.5963]])

    assert torch.allclose(input=result,
                          other=rights,
                          rtol=1e-04,
                          atol=1e-04,
                          equal_nan=True)


def test_stochastic_binarized_linear_non_bias():
    pytorch_lightning.seed_everything(777)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    inputs = torch.randn((3, 3))
    weights = torch.randn((3, 3))

    result = binary_linear(inputs,
                           weights,
                           None,
                           "stochastic")

    rights = torch.tensor([[1.8804,  1.4966,  2.6833],
                           [0.8230, -2.1745, -0.2879],
                           [1.5963, -4.2308, -1.2886]])

    assert torch.allclose(input=result,
                          other=rights,
                          rtol=1e-04,
                          atol=1e-04,
                          equal_nan=True)


def test_stochastic_binarized_linear_bias():
    pytorch_lightning.seed_everything(777)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    inputs = torch.randn((3, 3))
    weights = torch.randn((3, 3))

    result = binary_linear(inputs,
                           weights,
                           torch.tensor([1.]),
                           "stochastic")

    rights = torch.tensor([[2.8804,  2.4966,  3.6833],
                           [1.8230, -1.1745,  0.7121],
                           [2.5963, -3.2308, -0.2886]])

    assert torch.allclose(input=result,
                          other=rights,
                          rtol=1e-04,
                          atol=1e-04,
                          equal_nan=True)

# TODO. Gradient Check
