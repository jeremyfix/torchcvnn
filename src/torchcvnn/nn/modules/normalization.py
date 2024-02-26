# MIT License

# Copyright (c) 2024 Jeremy Fix

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

# Standard imports
from typing import Union, List
import math

# External imports
import torch
from torch import Size
import torch.nn as nn
import torch.nn.init as init

# Local imports
import torchcvnn.nn.modules.batchnorm as bn

_shape_t = Union[int, List[int], Size]


class LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = torch.complex64,
    ) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = torch.nn.parameter.Parameter(
                torch.empty(normalized_shape + (2, 2), device=device)
            )
            if self.bias:
                self.bias = torch.nn.parameter.Parameter(
                    torch.empty(normalized_shape, device=device, dtype=dtype)
                )
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            if self.elementwise_affine:
                # Initialize all the weights to zeros
                init.zeros_(self.weight)
                # And then fill in the diagonal with 1/sqrt(2)
                # w is C, 2, 2
                self.weight[:, 0, 0] = 1 / math.sqrt(2.0)
                self.weight[:, 1, 1] = 1 / math.sqrt(2.0)
                # Initialize all the biases to zero
                init.zeros_(self.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        pass
