from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F


class BinarizedConv2d(torch.autograd.Function):
    r"""
    Binary Operation에 대한 커스텀 Forward/Backward 정의
    Binary Operation Method는 `Deterministic`과 `Stochastic`으로 구분됨
    Refers:
        1). Custom Operation : https://pytorch.org/docs/stable/notes/extending.html
        2). Binary Operation Methods : https://arxiv.org/pdf/1511.00363.pdf
    """

    @staticmethod
    def forward(
        ctx: object,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        mode: str = "deterministic",
    ) -> torch.Tensor:
        r"""
        Binary forward operation을 정의한다.

        Note:
            forward는 binarized wegiths를 이용하지만, backward는 real-value weights를 이용한다.
            torch.nn.functional.conv2d를 참고하였다.
            `Deterministic`과 `Stochastic`을 별도로 구현한다.
            Refs: https://pytorch.org/docs/stable/nn.functional.html#conv2d

        Args:
            ctx         (object): forward/backward간 정보를 공유하기위한 context 정보
            input       (torch.Tensor): :math:`(N, *, in\_features)` where `*` means any number of additional dimensions
            weight      (torch.Tensor): :math:`(out\_features, in\_features)`
            bias        (Optional[torch.Tensor]): :math:`(out\_features)`
            stride      (Union[int, Tuple[int, int]]): the stride of the convolving kernel. Can be a single number or a tuple `(sH, sW)`. Default: 1
            padding     (Union[int, Tuple[int, int]]): implicit paddings on both sides of the input. Can be a single number or a tuple `(padH, padW)`. Default: 0
            dilation    (Union[int, Tuple[int, int]]): the spacing between kernel elements. Can be a single number or a tuple `(dH, dW)`. Default: 1
            groups:     (int): split input into groups, :math:`\text{in\_channels}` should be divisible by the number of groups. Default: 1
            mode        (str): `Deterministic`, `Stochastic` Method를 명시한다.

        Returns:
            (torch.Tensor) : binarized weights를 forwarding한 결과
        """

        # 별도로 Backward를 정의하므로 연산을 Computational Graph에 추가하지 않는다.
        with torch.no_grad():
            if mode == "deterministic":
                # torch.sign()함수는 Ternary로 Quantization 된다. [-1, 0, 1]
                # 따라서 `0`에 대한 별도 처리를 해야 Binary weight를 가질 수 있다.
                binarized_weight = weight.sign()
                binarized_weight[binarized_weight == 0] = 1.0
            elif mode == "stochastic":
                # weights를 sigmoid 입력으로 넣어 이를 확률값으로 변환한다. `sigmoid(weights)`
                # 해당 확률값을 이용하여 [-1, 1]을 생성한다.
                # +1 if w >= p, where p = sigmoid(w)
                # -1 else 1 - p

                # binarized probability를 먼저 구한다.
                # [0, 1]사이의 값을 갖는 데이터에서 uniform 확률 분포로 데이터를 샘플링한다.
                # sampling된 값들이 p값 이상을 갖으면 1, 그렇지 않으면 -1로 정의한다.
                binarized_probability = torch.sigmoid(weight)
                uniform_matrix = torch.empty(binarized_probability.shape).uniform_(0, 1)
                uniform_matrix = uniform_matrix.to(weight.device)
                binarized_weight = (binarized_probability >= uniform_matrix).type(
                    torch.float32
                )
                binarized_weight[binarized_weight == 0] = -1.0
            else:
                raise RuntimeError(f"{mode} not supported")

        with torch.no_grad():
            output = F.conv2d(
                input, binarized_weight, bias, stride, padding, dilation, groups
            )

        # Save input, binarized weight, bias in context object
        ctx.save_for_backward(input, binarized_weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups

        return output

    @staticmethod
    def backward(ctx: object, grad_output: Any) -> Tuple[Optional[torch.Tensor]]:
        r"""
        Binary backward operation을 정의한다.

        Note:
            forward는 binarized wegiths를 이용하지만, backward는 real-value weights를 이용한다.

        Args:
            ctx         (object): forward/backward간 정보를 공유하기위한 context 정보
            grad_output (Any): Compuational graph를 통해서 들어오는 gradient정보를 받는다.

        Returns:
            (torch.Tensor) : Computational graph 앞으로 보내기위한 gradient 정보
        """
        input, binarized_weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups
        grad_input = grad_weight = grad_bias = None

        with torch.no_grad():
            if ctx.needs_input_grad[0]:
                grad_input = torch.nn.grad.conv2d_input(
                    input.shape,
                    binarized_weight,
                    grad_output,
                    stride,
                    padding,
                    dilation,
                    groups,
                )
            if ctx.needs_input_grad[1]:
                grad_weight = torch.nn.grad.conv2d_weight(
                    input,
                    binarized_weight.shape,
                    grad_output,
                    stride,
                    padding,
                    dilation,
                    groups,
                )

            if bias is not None and ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum((0, 2, 3)).squeeze(0)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None


binarized_conv2d = BinarizedConv2d.apply
