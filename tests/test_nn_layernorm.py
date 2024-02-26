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
import time

# External imports
import torch

# Local imports
import torchcvnn.nn as c_nn
import torchcvnn.nn.modules.batchnorm as bn


def test_layernorm_real():
    # NLP Example
    batch, sentence_length, embedding_dim = 20, 5, 10
    embedding = torch.randn(batch, sentence_length, embedding_dim)
    layer_norm = torch.nn.LayerNorm(embedding_dim)
    layer_norm.eval()
    # Activate module
    out = layer_norm(embedding)

    assert out.shape == embedding.shape
    mus = out.mean(axis=-1)
    assert torch.allclose(mus, torch.zeros_like(mus), atol=1e-6)
    stds = out.std(axis=-1)
    assert torch.allclose(stds, torch.ones_like(stds), atol=1e-1)

    # Image Example
    N, C, H, W = 20, 5, 10, 10
    x = torch.randn(N, C, H, W)
    # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
    # as shown in the image below
    layer_norm = torch.nn.LayerNorm([C, H, W])
    layer_norm.eval()
    out = layer_norm(x)

    mus = out.view(N, -1).mean(axis=-1)
    assert torch.allclose(mus, torch.zeros_like(mus), atol=1e-6)
    stds = out.view(N, -1).std(axis=-1)
    assert torch.allclose(stds, torch.ones_like(stds), atol=1e-1)


def test_layernorm():
    # NLP Example
    batch, sentence_length, embedding_dim = 20, 5, 10
    embedding = torch.randn(
        (batch, sentence_length, embedding_dim), dtype=torch.complex64
    )
    layer_norm = c_nn.LayerNorm(embedding_dim)
    layer_norm.eval()
    # Activate module
    out = layer_norm(embedding)

    assert out.shape == embedding.shape

    mus = out.mean(axis=-1)
    assert torch.allclose(mus, torch.zeros_like(mus), atol=1e-6)

    out_real = torch.view_as_real(out)
    print(out_real.shape)
    covs = bn.batch_cov(out_real)

    stds = out.std(axis=-1)
    assert torch.allclose(stds, torch.ones_like(stds), atol=1e-1)

    # Image Example
    N, C, H, W = 20, 5, 10, 10
    x = torch.randn((N, C, H, W), dtype=torch.complex64)
    # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
    # as shown in the image below
    layer_norm = c_nn.LayerNorm([C, H, W])
    layer_norm.eval()
    # Activate module
    out = layer_norm(x)

    # # Compute the variance/covariance
    # xc = output.transpose(0, 1).reshape(C, -1)  # C, BxHxW
    # mus = xc.mean(axis=-1)  # 16 means
    # xc_real = torch.view_as_real(xc)
    # covs = bn.batch_cov(xc_real)  # 16 covariances matrices

    # # All the mus must be 0's
    # # For some reasons, this is not exactly 0
    # assert torch.allclose(mus, torch.zeros_like(mus), atol=1e-7)
    # # All the covs must be identities
    # id_cov = 0.5 * torch.eye(2).tile(C, 1, 1)
    # assert torch.allclose(covs, id_cov)


if __name__ == "__main__":
    test_layernorm_real()
    test_layernorm()
