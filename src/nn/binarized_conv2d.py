from typing import Optional, Tuple, Union

import torch
from torch.nn.modules.utils import _pair
from src.ops.binarized_conv2d import binarized_conv2d


class BinarizedConv2d(torch.nn.Conv2d):
    r"""Applies a 2D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\text{in}}, H, W)` and output :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`
    can be precisely described as:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)


    where :math:`\star` is the valid 2D `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`H` is a height of input planes in pixels, and :math:`W` is
    width in pixels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = (0, 0),
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: Optional[torch.Tensor] = None,
        padding_mode: str = "zeros",
        mode: str = "deterministic",
    ) -> torch.nn.Conv2d:
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.mode = mode

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.clipping()
        return binarized_conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.mode,
        )

    def clipping(self):
        """
        Since the binarization operation is not influenced by variations of the real-valued weights w 
        when its magnitude is beyond the binary values ±1, and since it is a common practice to bound weights (usually the weight vector) in order to regularize them, 
        we have chosen to clip the real-valued weights within the [−1, 1] interval right after the weight updates, 
        """
        with torch.no_grad():
            self.weight.clamp_(min=-1.0, max=1.0)
