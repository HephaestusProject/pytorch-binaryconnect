import os
import sys

import pytest
import pytorch_lightning
import torch

from src.ops.binarized_linear import BinarizedLinear, binarized_linear


@pytest.fixture(scope="module")
def fix_seed():
    pytorch_lightning.seed_everything(777)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


mode_test_case = [
    (
        torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        torch.tensor([[-1.0, 1.0, 1.0], [1.0, -0.8, 1.0], [1.0, -0.3, 1.0]]),
        None,
        "test",
    )
]


@pytest.mark.parametrize(
    "test_input, test_weight, test_bias, test_mode", mode_test_case
)
def test_supported_mode(fix_seed, test_input, test_weight, test_bias, test_mode):
    with pytest.raises(RuntimeError):
        binarized_linear(test_input, test_weight, test_bias, test_mode)


forward_test_case = [
    # (test_input, test_weight, test_bias, test_mode, expected)
    (
        torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        torch.tensor([[-1.0, 1.0, 1.0], [1.0, -0.8, 1.0], [1.0, -0.3, 1.0]]),
        None,
        "deterministic",
        torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
    ),
    (
        torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        torch.tensor([[-1.0, 1.0, 1.0], [1.0, -0.8, 1.0], [1.0, -0.3, 1.0]]),
        torch.tensor([1.0]),
        "deterministic",
        torch.tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]),
    ),
    (
        torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        torch.tensor([[-1.0, 1.0, 1.0], [1.0, -0.8, 1.0], [1.0, -0.3, 1.0]]),
        None,
        "stochastic",
        torch.tensor([[3.0, -1.0, -1.0], [3.0, -1.0, -1.0], [3.0, -1.0, -1.0]]),
    ),
    (
        torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        torch.tensor([[-1.0, 1.0, 1.0], [1.0, -0.8, 1.0], [1.0, -0.3, 1.0]]),
        torch.tensor([1.0]),
        "stochastic",
        torch.tensor([[4.0, 0.0, 0.0], [4.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
    ),
]


@pytest.mark.parametrize(
    "test_input, test_weight, test_bias, test_mode, expected", forward_test_case
)
def test_forward(fix_seed, test_input, test_weight, test_bias, test_mode, expected):
    assert torch.allclose(
        input=binarized_linear(test_input, test_weight, test_bias, test_mode),
        other=expected,
        rtol=1e-04,
        atol=1e-04,
        equal_nan=True,
    )


indirectly_backward_test_case = [
    # (test_input, test_weight, test_bias, test_mode, expected_weight_grad, expected_input_grad)
    (
        torch.tensor([[1.0, 1.0, 1.0]], requires_grad=True),
        torch.tensor([[-0.8, -0.8, 0.3]], requires_grad=True),
        None,
        "deterministic",
        torch.tensor([[1.0, 1.0, 1.0]]),
        torch.tensor([[-1.0, -1.0, 1.0]]),
    ),
    (
        torch.tensor([[1.0, 1.0, 1.0]], requires_grad=True),
        torch.tensor([[1.0, -0.8, 0.3]], requires_grad=True),
        torch.tensor([1]),
        "deterministic",
        torch.tensor([[1.0, 1.0, 1.0]]),
        torch.tensor([[1.0, -1.0, 1.0]]),
    ),
    (
        torch.tensor([[1.0, 1.0, 1.0]], requires_grad=True),
        torch.tensor([[1.0, -0.8, 0.3]], requires_grad=True),
        None,
        "stochastic",
        torch.tensor([[1.0, 1.0, 1.0]]),
        torch.tensor([[-1.0, -1.0, -1.0]]),
    ),
    (
        torch.tensor([[1.0, 1.0, 1.0]], requires_grad=True),
        torch.tensor([[1.0, -0.8, 0.3]], requires_grad=True),
        torch.tensor([1]),
        "deterministic",
        torch.tensor([[1.0, 1.0, 1.0]]),
        torch.tensor([[1.0, -1.0, 1.0]]),
    ),
]


@pytest.mark.parametrize(
    "test_input, test_weight, test_bias, test_mode, expected_weight_grad, expected_input_grad",
    indirectly_backward_test_case,
)
def test_backward_indirectly(
    fix_seed,
    test_input,
    test_weight,
    test_bias,
    test_mode,
    expected_weight_grad,
    expected_input_grad,
):

    binarized_linear(test_input, test_weight, test_bias, test_mode).backward()

    assert torch.allclose(
        input=test_input.grad,
        other=expected_input_grad,
        rtol=1e-04,
        atol=1e-04,
        equal_nan=True,
    )

    assert torch.allclose(
        input=test_weight.grad,
        other=expected_weight_grad,
        rtol=1e-04,
        atol=1e-04,
        equal_nan=True,
    )


directly_backward_test_case = [
    # (saved_tensors, needs_input_grad, grad_output, expected_weight_grad, expected_input_grad, expected_bias_grad)
    (
        (
            torch.tensor(
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], requires_grad=True
            ),
            torch.tensor([[1.0, -1.0, 1.0]]),
            torch.tensor([1]),
        ),
        (True, True, True, False),
        torch.tensor([[1.0], [1.0], [1.0]]),
        torch.tensor([[3.0, 3.0, 3.0]]),
        torch.tensor([[1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, -1.0, 1.0]]),
        torch.tensor(3.0),
    )
]


@pytest.mark.parametrize(
    "saved_tensors, needs_input_grad, grad_output, expected_weight_grad, expected_input_grad, expected_bias_grad",
    directly_backward_test_case,
)
def test_backward_directly(
    fix_seed,
    saved_tensors,
    needs_input_grad,
    grad_output,
    expected_weight_grad,
    expected_input_grad,
    expected_bias_grad,
):
    class CTX:
        def __init__(self, saved_tensors, needs_input_grad):
            self.saved_tensors = saved_tensors
            self.needs_input_grad = needs_input_grad

    ctx = CTX(saved_tensors, needs_input_grad)

    input_grad, weight_grad, bias_grad, _ = BinarizedLinear.backward(ctx, grad_output)

    assert torch.allclose(
        input=input_grad,
        other=expected_input_grad,
        rtol=1e-04,
        atol=1e-04,
        equal_nan=True,
    )

    assert torch.allclose(
        input=weight_grad,
        other=expected_weight_grad,
        rtol=1e-04,
        atol=1e-04,
        equal_nan=True,
    )

    assert torch.allclose(
        input=bias_grad,
        other=expected_bias_grad,
        rtol=1e-04,
        atol=1e-04,
        equal_nan=True,
    )
