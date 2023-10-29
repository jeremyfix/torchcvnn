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


class IndependentRealImag(nn.Module):
    """
    Generic module to apply a real valued activation function independently
    on both the real and imaginary part

    Arguments:
        fact: A nn.Module name of a real valued activation function
    """

    def __init__(self, fact: nn.Module):
        super().__init__()
        self.act_real = fact()
        self.act_imag = fact()

    def forward(self, z: torch.tensor) -> torch.tensor:
        """
        Performs the forward pass

        Arguments:
            z: the input tensor on which to apply the activation function
        """
        return self.act_real(z.real) + self.act_imag(z.imag) * 1j


class CReLU(IndependentRealImag):
    """
    Applies a ReLU independently on both the real and imaginary parts

    $$
    CReLU(z) = ReLU(Re[z]) + ReLU(Im[z])j
    $$

    Only the quadrant where both `Re[z]` and `Im[z]` are negative is projected to
    $0$. Otherwise either the real and/or the imaginary part is preserved.

    """

    def __init__(self) -> None:
        super().__init__(nn.ReLU)


class CPReLU(IndependentRealImag):
    """
    Applies a PReLU independently on both the real and imaginary parts

    $$
    CPReLU(z) = PReLU(Re[z]) + PReLU(Im[z])j
    $$
    """

    def __init__(self) -> None:
        super().__init__(nn.PReLU)


class CELU(IndependentRealImag):
    """
    Applies a ELU independently on both the real and imaginary parts

    Not to confuse with `torch.nn.CELU`. For the complex equivalent of
    `torch.nn.CELU`, see [torchcvnn.nn.modules.activation.CCELU][]

    $$
    CELU(z) = ELU(Re[z]) + ELU(Im[z])j
    $$
    """

    def __init__(self) -> None:
        super().__init__(nn.ELU)


class CCELU(IndependentRealImag):
    """
    Applies a CELU independently on both the real and imaginary parts

    $$
    CCELU(z) = CELU(Re[z]) + CELU(Im[z])j
    $$
    """

    def __init__(self) -> None:
        super().__init__(nn.CELU)


class CGELU(IndependentRealImag):
    """
    Applies a GELU independently on both the real and imaginary parts

    $$
    CGELU(z) = GELU(Re[z]) + GELU(Im[z])j
    $$
    """

    def __init__(self) -> None:
        super().__init__(nn.GELU)


class zReLU(nn.Module):
    r"""
    Applies a zReLU

    $$
    zRELU(z) = \begin{cases} 0 & \mbox{if } Re[z] > 0 \mbox{ and } Im[z] > 0\\ z & \mbox{otherwise}  \end{cases}
    $$

    All the quadrant where both `Re[z]` and `Im[z]` are non negative are
    projected to $0$. In other words, only one quadrant is preserved.
    """

    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor):
        """
        Performs the forward pass.

        Arguments:
            z: the input tensor on which to apply the activation function
        """
        pos_real = z.real > 0
        pos_img = z.imag > 0
        return z * pos_real * pos_img
