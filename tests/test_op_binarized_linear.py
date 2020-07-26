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
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], requires_grad=True
    )

    weights = torch.tensor([[1.0, -0.8, 0.3]], requires_grad=True)

    result = binary_linear(inputs, weights, None, "deterministic")

    result.sum().backward()

    target_backward_op_weights_grad = torch.tensor([[3., 3., 3.]])
    target_backward_op_inputs_grad = torch.tensor([[1., -1.,  1.],
                                                   [1., -1.,  1.],
                                                   [1., -1.,  1.]])

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
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], requires_grad=True
    )

    weights = torch.tensor([[1.0, -0.8, 0.3]], requires_grad=True)
    bias = torch.tensor([1])

    result = binary_linear(inputs, weights, bias, "deterministic")
    result.sum().backward()

    target_backward_op_weights_grad = torch.tensor([[3., 3., 3.]])
    target_backward_op_inputs_grad = torch.tensor(
        [
            [1., -1.,  1.],
            [1., -1.,  1.],
            [1., -1.,  1.]
        ]
    )

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

    inputs = torch.tensor(
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], requires_grad=True
    )
    weights = torch.tensor([[1.0, -0.8, 0.3]], requires_grad=True)

    result = binary_linear(inputs, weights, None, "stochastic")
    result.sum().backward()

    print(weights.grad)
    print(inputs.grad)

    target_backward_op_weights_grad = torch.tensor([[3., 3., 3.]])
    target_backward_op_inputs_grad = torch.tensor(
        [
            [-1., -1., -1.],
            [-1., -1., -1.],
            [-1., -1., -1.]
        ]
    )

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

    inputs = torch.tensor(
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], requires_grad=True
    )

    weights = torch.tensor([[1.0, -0.8, 0.3]], requires_grad=True)
    bias = torch.tensor([1])

    result = binary_linear(inputs, weights, bias, "stochastic")
    result.sum().backward()

    target_backward_op_weights_grad = torch.tensor([[3., 3., 3.]])
    target_backward_op_inputs_grad = torch.tensor(
        [
            [1., -1.,  1.],
            [1., -1.,  1.],
            [1., -1.,  1.]
        ]
    )

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
