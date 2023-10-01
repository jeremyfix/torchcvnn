import torch
import torch.nn as nn


class MaxPool2d(nn.Module):
    r"""Applies a 2D max pooling on the module of the input signal

    In the simplest case, the output value of the layer with input size $(N, C, H, W)$, output $(N, C, H_{out}, W_{out})$ and `kernel_size` `(kH, kW)` can be precisely described as:

    $$
    \begin{aligned}
    out(N_i, C_j, h, w) ={} & \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1} \\
                            & \text{|input|}(N_i, C_j, \text{stride[0]} \times h + m,
                                                   \text{stride[1]} \times w + n)
    \end{aligned}
    $$

    """

    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        dilation=1,
        ceil_mode=False,
        return_indices=False,
    ) -> None:
        super().__init__()
        self.return_indices = return_indices
        self.m = nn.MaxPool2d(
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode=ceil_mode,
            return_indices=True,
        )

    def forward(self, z: torch.Tensor):
        _, indices = self.m(torch.abs(z))

        if self.return_indices:
            return z.flatten()[indices], indices
        else:
            return z.flatten()[indices]


class AvgPool2d(nn.Module):
    """
    Implementation of torch.nn.AvgPool2d for complex numbers.
    Apply AvgPool2d on the real and imaginary part.
    Returns complex values associated to the AvgPool2dresults.
    """

    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=None,
    ) -> None:
        super().__init__()
        if type(kernel_size) == int:
            self.kernel_size = [kernel_size] * 2 + [1]
        elif type(kernel_size) == tuple:
            if len(kernel_size) < 3:
                self.kernel_size = [kernel_size] + [1]
            else:
                self.kernel_size = kernel_size

        if type(stride) == int:
            self.stride = [stride] * 2 + [1]
        elif type(stride) == tuple:
            if len(stride) < 3:
                self.stride = [stride] + [1]
            else:
                self.stride = stride

        self.m = torch.nn.AvgPool3d(
            self.kernel_size,
            self.stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
        )

    def forward(self, z: torch.Tensor):
        return torch.view_as_complex(self.m(torch.view_as_real(z)))
