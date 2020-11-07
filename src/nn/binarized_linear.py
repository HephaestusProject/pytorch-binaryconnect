from typing import Optional

import torch

from src.ops.binarized_linear import binarized_linear


class BinarizedLinear(torch.nn.Linear):
    r"""Applies a Binarized linear transformation to the incoming data: :math:`y = xA_{b}^{T} + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::
        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, mode: str = "deterministic",
    ) -> torch.nn.Linear:
        super().__init__(in_features, out_features, bias)
        self.mode = mode

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.clipping()
        if self.bias is not None:
            return binarized_linear(input, self.weight, self.bias)
        return binarized_linear(input, self.weight)

    def clipping(self):
        """
        Since the binarization operation is not influenced by variations of the real-valued weights w 
        when its magnitude is beyond the binary values ±1, and since it is a common practice to bound weights (usually the weight vector) in order to regularize them, 
        we have chosen to clip the real-valued weights within the [−1, 1] interval right after the weight updates, 
        """
        with torch.no_grad():
            self.weight.clamp_(min=-1.0, max=1.0)
