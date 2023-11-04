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

# Local imports
import torchcvnn.nn as c_nn
import torchcvnn.nn.modules.batchnorm as bn


def test_batchnorm2d():
    B, C, H, W = 20, 16, 50, 100
    m = c_nn.BatchNorm2d(C)

    x = torch.randn((B, C, H, W), dtype=torch.complex64)
    output = m(x)

    # Compute the variance/covariance
    xc = output.transpose(0, 1).reshape(C, -1)  # C, BxHxW
    mus = xc.mean(axis=-1)  # 16 means
    xc_real = torch.view_as_real(xc)
    covs = bn.batch_cov(xc_real)  # 16 covariances matrices

    # All the mus must be 0's
    assert torch.allclose(mus, torch.zeros_like(mus))
    # All the covs must be identities
    # TODO:


if __name__ == "__main__":
    test_batchnorm2d()
