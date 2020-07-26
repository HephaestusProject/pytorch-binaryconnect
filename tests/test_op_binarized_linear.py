import os
import sys

import pytest
import pytorch_lightning
import torch

from binaryconnect.ops.binarized_linear import binary_linear


@pytest.fixture(scope="module")
def fix_seed():
    pytorch_lightning.seed_everything(777)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test_not_support_mode_binarized_linear(fix_seed):
    inputs = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    weights = torch.tensor(
        [[-1.0, 1.0, 1.0], [1.0, -0.8, 1.0], [1.0, -0.3, 1.0]])

    with pytest.raises(RuntimeError):
        binary_linear(inputs, weights, None, "test")


def test_foward_op_in_non_bias_deterministic_binarized_linear(fix_seed):
    inputs = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

    weights = torch.tensor(
        [[-1.0, 1.0, 1.0], [1.0, -0.8, 1.0], [1.0, -0.3, 1.0]])

    result = binary_linear(inputs, weights, None, "deterministic")

    target_forward_op = torch.tensor(
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    )

    assert torch.allclose(
        input=result, other=target_forward_op, rtol=1e-04, atol=1e-04, equal_nan=True
    )


def test_foward_op_in_bias_deterministic_binarized_linear(fix_seed):

    inputs = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

    weights = torch.tensor(
        [[-1.0, 1.0, 1.0], [1.0, -0.8, 1.0], [1.0, -0.3, 1.0]])

    result = binary_linear(
        inputs, weights, torch.tensor([1.0]), "deterministic")

    target_forward_op = torch.tensor(
        [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]
    )

    assert torch.allclose(
        input=result, other=target_forward_op, rtol=1e-04, atol=1e-04, equal_nan=True
    )


def test_foward_op_in_non_bias_stochastic_binarized_linear(fix_seed):
    inputs = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

    weights = torch.tensor(
        [[-1.0, 1.0, 1.0], [1.0, -0.8, 1.0], [1.0, -0.3, 1.0]])

    result = binary_linear(inputs, weights, None, "stochastic")

    target_forward_op = torch.tensor(
        [[3.0, -1.0, -1.0], [3.0, -1.0, -1.0], [3.0, -1.0, -1.0]]
    )

    assert torch.allclose(
        input=result, other=target_forward_op, rtol=1e-04, atol=1e-04, equal_nan=True
    )


def test_foward_op_in_bias_stochastic_binarized_linear(fix_seed):

    inputs = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

    weights = torch.tensor(
        [[-1.0, 1.0, 1.0], [1.0, -0.8, 1.0], [1.0, -0.3, 1.0]])

    result = binary_linear(inputs, weights, torch.tensor([1.0]), "stochastic")

    target_forward_op = torch.tensor(
        [[4.0, 0.0, 0.0], [4.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
    )

    assert torch.allclose(
        input=result, other=target_forward_op, rtol=1e-04, atol=1e-04, equal_nan=True
    )


# Gradient Check
# loss  = (result - target)^2
# A = frac{partial{loss}}{partial{result}}
#   = 2(result - target) / N(`size`)
# B = frac{partial{result}}{partial{weights}}
#   = inputs
# C = frac{partial{result}}{partial{inputs}}
#   = binarized_weights

# gradient of weights   = A * B
# gradient of inputs    = A * C


def test_backward_op_in_non_bias_deterministic_binarized_linear(fix_seed):
    inputs = torch.tensor(
        [[1.0, 1.0, 1.0]], requires_grad=True
    )
    weights = torch.tensor([[-0.8, -0.8, 0.3]], requires_grad=True)

    binary_linear(inputs, weights, None, "deterministic").backward()

    target_backward_op_weights_grad = torch.tensor([[1., 1., 1.]])
    target_backward_op_inputs_grad = torch.tensor([[-1., -1.,  1.]])

    assert torch.allclose(
        input=weights.grad,
        other=target_backward_op_weights_grad,
        rtol=1e-04,
        atol=1e-04,
        equal_nan=True,
    )

    assert torch.allclose(
        input=inputs.grad,
        other=target_backward_op_inputs_grad,
        rtol=1e-04,
        atol=1e-04,
        equal_nan=True,
    )


def test_backward_op_in_bias_deterministic_binarized_linear(fix_seed):

    inputs = torch.tensor(
        [[1.0, 1.0, 1.0]], requires_grad=True
    )

    weights = torch.tensor([[1.0, -0.8, 0.3]], requires_grad=True)
    bias = torch.tensor([1])

    binary_linear(inputs, weights, bias, "deterministic").backward()

    target_backward_op_weights_grad = torch.tensor([[1., 1., 1.]])
    target_backward_op_inputs_grad = torch.tensor([[1., -1.,  1.]])

    assert torch.allclose(
        input=weights.grad,
        other=target_backward_op_weights_grad,
        rtol=1e-04,
        atol=1e-04,
        equal_nan=True,
    )

    assert torch.allclose(
        input=inputs.grad,
        other=target_backward_op_inputs_grad,
        rtol=1e-04,
        atol=1e-04,
        equal_nan=True,
    )


def test_backward_op_in_non_bias_stochastic_binarized_linear(fix_seed):

    inputs = torch.tensor([[1.0, 1.0, 1.0]], requires_grad=True)
    weights = torch.tensor([[1.0, -0.8, 0.3]], requires_grad=True)

    binary_linear(inputs, weights, None, "stochastic").backward()

    target_backward_op_weights_grad = torch.tensor([[1., 1., 1.]])
    target_backward_op_inputs_grad = torch.tensor([[-1., -1., -1.]])

    assert torch.allclose(
        input=weights.grad,
        other=target_backward_op_weights_grad,
        rtol=1e-04,
        atol=1e-04,
        equal_nan=True,
    )

    assert torch.allclose(
        input=inputs.grad,
        other=target_backward_op_inputs_grad,
        rtol=1e-04,
        atol=1e-04,
        equal_nan=True,
    )


def test_backward_op_in_bias_stochastic_binarized_linear(fix_seed):

    inputs = torch.tensor([[1.0, 1.0, 1.0]], requires_grad=True)

    weights = torch.tensor([[1.0, -0.8, 0.3]], requires_grad=True)
    bias = torch.tensor([1])

    binary_linear(inputs, weights, bias, "stochastic").backward()

    target_backward_op_weights_grad = torch.tensor([[1., 1., 1.]])
    target_backward_op_inputs_grad = torch.tensor([[1., -1.,  1.]])

    assert torch.allclose(
        input=weights.grad,
        other=target_backward_op_weights_grad,
        rtol=1e-04,
        atol=1e-04,
        equal_nan=True,
    )

    assert torch.allclose(
        input=inputs.grad,
        other=target_backward_op_inputs_grad,
        rtol=1e-04,
        atol=1e-04,
        equal_nan=True,
    )


def test_backward(fix_seed):
    from binaryconnect.ops.binarized_linear import BinaryLinear

    class CTX:
        saved_tensors = (torch.tensor([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]], requires_grad=True),
                         torch.tensor([[1., -1.,  1.]]),
                         None)
        needs_input_grad = (True, True, False, False)

    ctx = CTX()
    grad_output = torch.tensor([[1.], [1.], [1.]])
    _, grad, _, _ = BinaryLinear.backward(ctx, grad_output)

    target_backward_op_weights_grad = torch.tensor([[3., 3., 3.]])

    assert torch.allclose(
        input=grad,
        other=target_backward_op_weights_grad,
        rtol=1e-04,
        atol=1e-04,
        equal_nan=True,
    )
