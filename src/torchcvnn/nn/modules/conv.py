# MIT License

# Copyright (c) 2023 Jérémie Levi, Victor Dhédin, Jeremy Fix

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# External imports
import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t


class ConvTranspose2d(nn.Module):
    """
    Implementation of torch.nn.Conv2dTranspose for complex numbers.
    Apply Conv2dTranspose on real and imaginary part of the complex number.

    The parameters are the same than from the upstream pytorch layer:

    Arguments:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): ``dilation * (kernel_size - 1) - padding`` zero-padding will be added to both sides of each dimension in the input. Default: 0
        output_padding (int or tuple, optional): Additional size added to one side of each dimension in the output shape. Default: 0
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        padding_mode (str, optional): ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        output_padding: _size_2_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_2_t = 1,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        self.m_real = torch.nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            padding_mode,
            device,
            dtype,
        )

        self.m_imag = torch.nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            padding_mode,
            device,
            dtype,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass, applying real valued ConvTranspose2d
        independently on both the real and imaginary parts of the input.
        """
        return torch.view_as_complex(
            torch.cat(
                (
                    torch.unsqueeze(self.m_real(z.real) - self.m_imag(z.imag), -1),
                    torch.unsqueeze(self.m_real(z.imag) + self.m_imag(z.real), -1),
                ),
                axis=-1,
            )
        )
