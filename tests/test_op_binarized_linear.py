import os
import sys
import pytest

import torch
import pytorch_lightning

current_file_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
if current_file_path not in sys.path:
    sys.path.append(current_file_path)
    from binaryconnect.ops.binarized_linear import binary_linear


def test_foward_op_in_non_bias_deterministic_binarized_linear():
    pytorch_lightning.seed_everything(777)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    inputs = torch.tensor([[1., 1., 1.],
                           [1., 1., 1.],
                           [1., 1., 1.]])

    weights = torch.tensor([[-1., 1., 1.],
                            [1., -.8, 1.],
                            [1., -.3, 1.]])

    result = binary_linear(inputs,
                           weights,
                           None,
                           "deterministic")

    target_forward_op = torch.tensor([[1., 1.,  1.],
                                      [1., 1.,  1.],
                                      [1., 1.,  1.]])

    assert torch.allclose(input=result,
                          other=target_forward_op,
                          rtol=1e-04,
                          atol=1e-04,
                          equal_nan=True)


def test_foward_op_in_bias_deterministic_binarized_linear():
    pytorch_lightning.seed_everything(777)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    inputs = torch.tensor([[1., 1., 1.],
                           [1., 1., 1.],
                           [1., 1., 1.]])

    weights = torch.tensor([[-1., 1., 1.],
                            [1., -.8, 1.],
                            [1., -.3, 1.]])

    result = binary_linear(inputs,
                           weights,
                           torch.tensor([1.]),
                           "deterministic")

    target_forward_op = torch.tensor([[2.,  2.,  2.],
                                      [2.,  2.,  2.],
                                      [2.,  2.,  2.]])

    assert torch.allclose(input=result,
                          other=target_forward_op,
                          rtol=1e-04,
                          atol=1e-04,
                          equal_nan=True)


def test_foward_op_in_non_bias_stochastic_binarized_linear():
    pytorch_lightning.seed_everything(777)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    inputs = torch.tensor([[1., 1., 1.],
                           [1., 1., 1.],
                           [1., 1., 1.]])

    weights = torch.tensor([[-1., 1., 1.],
                            [1., -.8, 1.],
                            [1., -.3, 1.]])

    result = binary_linear(inputs,
                           weights,
                           None,
                           "stochastic")

    target_forward_op = torch.tensor([[3., -1., -1.],
                                      [3., -1., -1.],
                                      [3., -1., -1.]])

    assert torch.allclose(input=result,
                          other=target_forward_op,
                          rtol=1e-04,
                          atol=1e-04,
                          equal_nan=True)


def test_foward_op_in_bias_stochastic_binarized_linear():
    pytorch_lightning.seed_everything(777)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    inputs = torch.tensor([[1., 1., 1.],
                           [1., 1., 1.],
                           [1., 1., 1.]])

    weights = torch.tensor([[-1., 1., 1.],
                            [1., -.8, 1.],
                            [1., -.3, 1.]])

    result = binary_linear(inputs,
                           weights,
                           torch.tensor([1.]),
                           "stochastic")

    target_forward_op = torch.tensor([[4., 0., 0.],
                                      [4., 0., 0.],
                                      [4., 0., 0.]])

    assert torch.allclose(input=result,
                          other=target_forward_op,
                          rtol=1e-04,
                          atol=1e-04,
                          equal_nan=True)

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


def test_backward_op_in_non_bias_deterministic_binarized_linear():
    pytorch_lightning.seed_everything(777)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    inputs = torch.tensor([[1., 1., 1.],
                           [1., 1., 1.],
                           [1., 1., 1.]], requires_grad=True)

    weights = torch.tensor([[1., -.8, .3]], requires_grad=True)

    target = torch.tensor([[0.],
                           [1.],
                           [0.]])

    result = binary_linear(inputs,
                           weights,
                           None,
                           "deterministic")

    loss = torch.nn.functional.mse_loss(result, target)
    loss.backward()

    # frac{partial{loss}}{partial{result}} = [0.66, 0, 0.66]
    # frac{partial{result}}{partial{weights}}   =   [[1., 1., 1.],
    #                                                [1., 1., 1.],
    #                                                [1., 1., 1.]]
    # frac{partial{result}}{partial{inputs}}    =   [[1.0, -1.0, 1.0]]

    target_backward_op_weights_grad = torch.tensor([[1.3333, 1.3333, 1.3333]])
    target_backward_op_inputs_grad = torch.tensor([[0.6667, -0.6667,  0.6667],
                                                   [0.0000, -0.0000,  0.0000],
                                                   [0.6667, -0.6667,  0.6667]])

    assert torch.allclose(input=weights.grad,
                          other=target_backward_op_weights_grad,
                          rtol=1e-04,
                          atol=1e-04,
                          equal_nan=True)

    assert torch.allclose(input=inputs.grad,
                          other=target_backward_op_inputs_grad,
                          rtol=1e-04,
                          atol=1e-04,
                          equal_nan=True)


def test_backward_op_in_bias_deterministic_binarized_linear():
    pytorch_lightning.seed_everything(777)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    inputs = torch.tensor([[1., 1., 1.],
                           [1., 1., 1.],
                           [1., 1., 1.]], requires_grad=True)

    weights = torch.tensor([[1., -.8, .3]], requires_grad=True)
    bias = torch.tensor([1])

    target = torch.tensor([[0.],
                           [1.],
                           [0.]])

    result = binary_linear(inputs,
                           weights,
                           bias,
                           "deterministic")

    loss = torch.nn.functional.mse_loss(result, target)
    loss.backward()

    # frac{partial{loss}}{partial{result}} = [[1.33], [0.66], [1.33]]
    # frac{partial{result}}{partial{weights}}   =   [[1., 1., 1.],
    #                                                [1., 1., 1.],
    #                                                [1., 1., 1.]]
    # frac{partial{result}}{partial{inputs}}    =   [[1.0, -1.0, 1.0]]

    target_backward_op_weights_grad = torch.tensor([[3.3333, 3.3333, 3.3333]])
    target_backward_op_inputs_grad = torch.tensor([[1.3333, -1.3333,  1.3333],
                                                   [0.6667, -0.6667,  0.6667],
                                                   [1.3333, -1.3333,  1.3333]])

    assert torch.allclose(input=weights.grad,
                          other=target_backward_op_weights_grad,
                          rtol=1e-04,
                          atol=1e-04,
                          equal_nan=True)

    assert torch.allclose(input=inputs.grad,
                          other=target_backward_op_inputs_grad,
                          rtol=1e-04,
                          atol=1e-04,
                          equal_nan=True)


def test_backward_op_in_non_bias_stochastic_binarized_linear():
    pytorch_lightning.seed_everything(777)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    inputs = torch.tensor([[1., 1., 1.],
                           [1., 1., 1.],
                           [1., 1., 1.]], requires_grad=True)

    weights = torch.tensor([[1., -.8, .3]], requires_grad=True)

    target = torch.tensor([[0.],
                           [1.],
                           [0.]])

    result = binary_linear(inputs,
                           weights,
                           None,
                           "stochastic")

    loss = torch.nn.functional.mse_loss(result, target)
    loss.backward()

    # frac{partial{loss}}{partial{result}} = [[0.66], [0.], [0.66]]
    # frac{partial{result}}{partial{weights}}   =   [[1., 1., 1.],
    #                                                [1., 1., 1.],
    #                                                [1., 1., 1.]]
    # frac{partial{result}}{partial{inputs}}    =   [[1.0, -1.0, 1.0]]

    target_backward_op_weights_grad = torch.tensor([[1.3333, 1.3333, 1.3333]])
    target_backward_op_inputs_grad = torch.tensor([[0.6667, -0.6667,  0.6667],
                                                   [0.0000, -0.0000,  0.0000],
                                                   [0.6667, -0.6667,  0.6667]])

    assert torch.allclose(input=weights.grad,
                          other=target_backward_op_weights_grad,
                          rtol=1e-04,
                          atol=1e-04,
                          equal_nan=True)

    assert torch.allclose(input=inputs.grad,
                          other=target_backward_op_inputs_grad,
                          rtol=1e-04,
                          atol=1e-04,
                          equal_nan=True)


def test_backward_op_in_bias_stochastic_binarized_linear():
    pytorch_lightning.seed_everything(777)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    inputs = torch.tensor([[1., 1., 1.],
                           [1., 1., 1.],
                           [1., 1., 1.]], requires_grad=True)

    weights = torch.tensor([[1., -.8, .3]], requires_grad=True)
    bias = torch.tensor([1])

    target = torch.tensor([[0.],
                           [1.],
                           [0.]])

    result = binary_linear(inputs,
                           weights,
                           bias,
                           "stochastic")

    loss = torch.nn.functional.mse_loss(result, target)
    loss.backward()

    # frac{partial{loss}}{partial{result}} = [[1.33], [0.66], [1.33]]
    # frac{partial{result}}{partial{weights}}   =   [[1., 1., 1.],
    #                                                [1., 1., 1.],
    #                                                [1., 1., 1.]]
    # frac{partial{result}}{partial{inputs}}    =   [[1.0, -1.0, 1.0]]

    target_backward_op_weights_grad = torch.tensor([[3.3333, 3.3333, 3.3333]])
    target_backward_op_inputs_grad = torch.tensor([[1.3333, -1.3333,  1.3333],
                                                   [0.6667, -0.6667,  0.6667],
                                                   [1.3333, -1.3333,  1.3333]])

    assert torch.allclose(input=weights.grad,
                          other=target_backward_op_weights_grad,
                          rtol=1e-04,
                          atol=1e-04,
                          equal_nan=True)

    assert torch.allclose(input=inputs.grad,
                          other=target_backward_op_inputs_grad,
                          rtol=1e-04,
                          atol=1e-04,
                          equal_nan=True)
