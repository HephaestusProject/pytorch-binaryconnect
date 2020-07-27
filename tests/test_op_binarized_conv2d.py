import os
import sys

import pytest
import pytorch_lightning
import torch

from src.ops.binarized_conv2d import (BinaryConv2d, binary_conv2d)


@pytest.fixture(scope="module")
def fix_seed():
    pytorch_lightning.seed_everything(777)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test_not_support_mode_binary_conv2d(fix_seed):

    inputs = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    weights = torch.tensor(
        [[-1.0, 1.0, 1.0], [1.0, -0.8, 1.0], [1.0, -0.3, 1.0]])
    bias = None
    stride = 1
    padding = 0
    dilation = 1
    groups = 1
    mode = "test"

    with pytest.raises(RuntimeError):
        binary_conv2d(inputs, weights, bias, stride,
                      padding, dilation, groups, mode)


def test_foward_op_in_non_bias_deterministic_binary_conv2d(fix_seed):

    inputs = torch.tensor(
        [[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]])

    weights = torch.tensor(
        [[[[-1.0, 1.0, 1.0], [1.0, -0.8, 1.0], [1.0, -0.3, 1.0]]]])
    bias = None
    stride = 1
    padding = 0
    dilation = 1
    groups = 1
    mode = "deterministic"

    result = binary_conv2d(inputs, weights, bias, stride,
                           padding, dilation, groups, mode)

    assert torch.allclose(
        input=result, other=torch.tensor([[[[3.]]]]), rtol=1e-04, atol=1e-04, equal_nan=True
    )


def test_foward_op_in_bias_deterministic_binary_conv2d(fix_seed):

    inputs = torch.tensor(
        [[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]])

    weights = torch.tensor(
        [[[[-1.0, 1.0, 1.0], [1.0, -0.8, 1.0], [1.0, -0.3, 1.0]]]])

    bias = torch.tensor([1.])
    stride = 1
    padding = 0
    dilation = 1
    groups = 1
    mode = "deterministic"

    result = binary_conv2d(inputs, weights, bias, stride,
                           padding, dilation, groups, mode)

    target_forward_op = torch.tensor(
        [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]
    )

    assert torch.allclose(
        input=result, other=torch.tensor([[[[4.]]]]), rtol=1e-04, atol=1e-04, equal_nan=True
    )


def test_foward_op_in_non_bias_stochastic_binary_conv2d(fix_seed):
    inputs = torch.tensor(
        [[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]])

    weights = torch.tensor(
        [[[[-1.0, 1.0, 1.0], [1.0, -0.8, 1.0], [1.0, -0.3, 1.0]]]])

    bias = None
    stride = 1
    padding = 0
    dilation = 1
    groups = 1
    mode = "stochastic"

    result = binary_conv2d(inputs, weights, bias, stride,
                           padding, dilation, groups, mode)

    assert torch.allclose(
        input=result, other=torch.tensor([[[[1.]]]]), rtol=1e-04, atol=1e-04, equal_nan=True
    )


def test_foward_op_in_bias_stochastic_binary_conv2d(fix_seed):

    inputs = torch.tensor(
        [[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]])

    weights = torch.tensor(
        [[[[-1.0, 1.0, 1.0], [1.0, -0.8, 1.0], [1.0, -0.3, 1.0]]]])

    bias = torch.tensor([1.])
    stride = 1
    padding = 0
    dilation = 1
    groups = 1
    mode = "stochastic"

    result = binary_conv2d(inputs, weights, bias, stride,
                           padding, dilation, groups, mode)

    assert torch.allclose(
        input=result, other=torch.tensor([[[[2.]]]]), rtol=1e-04, atol=1e-04, equal_nan=True
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

"""
def test_backward_op_in_non_bias_deterministic_binary_conv2d(fix_seed):
    inputs = torch.tensor(
        [[1.0, 1.0, 1.0]], requires_grad=True
    )
    weights = torch.tensor([[-0.8, -0.8, 0.3]], requires_grad=True)

    binary_conv2d(inputs, weights, None, "deterministic").backward()

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


def test_backward_op_in_bias_deterministic_binary_conv2d(fix_seed):

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


def test_backward_op_in_non_bias_stochastic_binary_conv2d(fix_seed):

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


def test_backward_op_in_bias_stochastic_binary_conv2d(fix_seed):

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

    class CTX:
        saved_tensors = (torch.tensor([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]], requires_grad=True),
                         torch.tensor([[1., -1.,  1.]]),
                         torch.tensor([1]))
        needs_input_grad = (True, True, True, False)

    ctx = CTX()
    grad_output = torch.tensor([[1.], [1.], [1.]])

    input_grad, weight_gard, bias_grad, _ = BinaryConv2d.backward(
        ctx, grad_output)

    target_backward_op_weights_grad = torch.tensor([[3., 3., 3.]])
    target_backward_op_inputs_grad = torch.tensor([[1., -1.,  1.],
                                                   [1., -1.,  1.],
                                                   [1., -1.,  1.]])
    target_backward_op_bias_grad = torch.tensor(3.)

    assert torch.allclose(
        input=weight_gard,
        other=target_backward_op_weights_grad,
        rtol=1e-04,
        atol=1e-04,
        equal_nan=True,
    )

    assert torch.allclose(
        input=input_grad,
        other=target_backward_op_inputs_grad,
        rtol=1e-04,
        atol=1e-04,
        equal_nan=True,
    )

    assert torch.allclose(
        input=bias_grad,
        other=target_backward_op_bias_grad,
        rtol=1e-04,
        atol=1e-04,
        equal_nan=True,
    )
"""
