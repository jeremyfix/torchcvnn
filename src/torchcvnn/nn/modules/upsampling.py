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


# Standard imports
from typing import Optional

# External imports
import torch
import torch.nn as nn
from torch.nn.common_types import _size_any_t, _ratio_any_t


class Upsample(nn.Module):

    """
    Works by applying independently the same upsampling to both the real and
    imaginary parts.

    Note:
        With pytorch 2.1, applying the nn.Upsample to a complex valued tensor raises
        an exception "compute_indices_weights_nearest" not implemented for 'ComplexFloat'
        So it basically splits the input tensors in its real and imaginery
        parts, applies nn.Upsample on both components and view them as complex.


    Arguments:
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int], optional):
            output spatial sizes
        scale_factor (float or Tuple[float] or Tuple[float, float] or Tuple[float, float, float], optional):
            multiplier for spatial size. Has to match input size if it is a tuple.
        mode (str, optional): the upsampling algorithm: one of ``'nearest'``,
            ``'linear'``, ``'bilinear'``, ``'bicubic'`` and ``'trilinear'``.
            Default: ``'nearest'``
        align_corners (bool, optional): if ``True``, the corner pixels of the input
            and output tensors are aligned, and thus preserving the values at
            those pixels. This only has effect when :attr:`mode` is
            ``'linear'``, ``'bilinear'``, ``'bicubic'``, or ``'trilinear'``.
            Default: ``False``
        recompute_scale_factor (bool, optional): recompute the scale_factor for use in the
            interpolation calculation. If `recompute_scale_factor` is ``True``, then
            `scale_factor` must be passed in and `scale_factor` is used to compute the
            output `size`. The computed output `size` will be used to infer new scales for
            the interpolation. Note that when `scale_factor` is floating-point, it may differ
            from the recomputed `scale_factor` due to rounding and precision issues.
            If `recompute_scale_factor` is ``False``, then `size` or `scale_factor` will
            be used directly for interpolation.
    """

    def __init__(
        self,
        size: Optional[_size_any_t] = None,
        scale_factor: Optional[_ratio_any_t] = None,
        mode: str = "nearest",
        align_corners: Optional[bool] = None,
        recompute_scale_factor: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.up_module = nn.Upsample(
            size, scale_factor, mode, align_corners, recompute_scale_factor
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Applies the forward pass
        """
        up_real = self.up_module(z.real).unsqueeze(-1)
        up_imag = self.up_module(z.imag).unsqueeze(-1)
        up_z = torch.cat((up_real, up_imag), axis=-1)
        return torch.view_as_complex(up_z)
