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
import torch.nn.init as init


def inv_sqrt_22_batch(M: torch.Tensor) -> torch.Tensor:
    """
    Computes the square root of the inverse of a tensor of shape [N, 2, 2]
    """
    N = M.shape[0]
    # [M, 1, 1]
    trace = torch.diagonal(M, dim1=-2, dim2=-1).sum(-1).reshape(N, 1, 1)
    det = torch.linalg.det(M).reshape(N, 1, 1)

    s = torch.sqrt(det)
    t = torch.sqrt(trace + 2 * s)
    # [M, 2, 2]
    M_inv = 1 / t * (M + s * torch.eye(2).tile(N, 1, 1))
    return M_inv


class BatchNorm(nn.Module):
    """
    BatchNorm for complex valued neural networks.

    As defined by Trabelsi et al. (2018)
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        device=None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': torch.complex64}
        super().__init__()
   
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.device = device

        if self.affine:
            self.weight = Parameter(torch.empty(num_features, **factory_kwargs))
            self.bias = Parameter(torch.empty(num_features, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.register_buffer('running_mean', torch.zeros(num_features, **factory_kwargs))
        self.register_buffer('running_var', torch.ones(num_features, **factory_kwargs))

        self.reset_parameters()

    def reset_running_stats(self) -> None:
        self.running_mean.zero_()
        self.running_var[]
        identity = torch.eye(2) / torch.sqrt(torch.tensor(2))
        self.gamma = torch.nn.Parameter(identity, requires_grad=True, device=device)
        self.beta = torch.nn.Parameter(
            torch.zeros(2), requires_grad=True, device=device
        )

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def init_running_avg(self, n_features):
        I = torch.eye(2) / torch.sqrt(torch.tensor(2))
        # [C, H, W, 2, 2]
        self.sigma_running = self.sigma = I.tile(n_features, 1, 1)
        # [C, H, W, 2]
        self.mu_running = self.mu = torch.zeros(1, 2).tile(n_features, 1)

    def update_mu_sigma(self, z: torch.Tensor):
        """
        Update mu and sigma matrix.
        """
        # [B, C, H, W, 2]
        B, C, H, W, _ = z.shape
        # [B, C*H*W, 2]
        z = z.reshape(B, C * H * W, 2)

        ### MEAN
        # [1, C*H*W, 2]
        self.mu = torch.mean(z, dim=0).unsqueeze(0)

        ### COV
        # [B, C*H*W, 2]
        x_centered = z - self.mu
        # [B*C*H*W, 2]
        x_centered_b = x_centered.reshape(B * C * H * W, 2)
        # [B*C*H*W, 2] = [B*C*H*W, 2, 1] @ [B*C*H*W, 1, 2]
        prods = torch.bmm(x_centered_b.unsqueeze(2), x_centered_b.unsqueeze(1))
        # [B, C*H*W, 2, 2]
        prods = prods.view(B, C * H * W, 2, 2)
        # [1, C*H*W, 2, 2]
        self.sigma = 1 / (B - 1) * prods.sum(dim=0)

        self.sigma_running = (
            1 - self.momentum
        ) * self.sigma_running + self.momentum * self.sigma
        self.mu_running = (
            1 - self.momentum
        ) * self.mu_running + self.momentum * self.mu

    def normalize(self, z: torch.Tensor):
        """
        Returns a normalized tensor of input z (complex) [B, C, H, W].
        Uses self.sigma as covariance matrix and self.mu as mean vector.
        Return:
            torch.Tensor [B*C*H*W, 2] (complex viewed as real)
        """
        B, C, H, W, _ = z.shape
        z = z.reshape(B, C * H * W, 2)
        # [C*H*W, 2, 2]
        sigma_inv_sqrt = self.inv_sqrt_22_batch(self.sigma)
        # [B, C*H*W, 2, 1]
        z_centered = (z - self.mu).reshape(B * C * H * W, 2).unsqueeze(-1)
        o = torch.bmm(
            sigma_inv_sqrt.tile(B, 1, 1).reshape(B * C * H * W, 2, 2), z_centered
        )
        return o

    def normalize_inference(self, z: torch.Tensor):
        """
        Returns a normalized tensor of input z (complex) [B, C, H, W].
        Uses self.sigma_running as covariance matrix and self.mu_running as mean vector.
        Return:
            torch.Tensor [B*C*H*W, 2] (complex viewed as real)
        """
        B, C, H, W, _ = z.shape
        # [C*H*W, 2, 2]
        sigma_inv_sqrt = self.inv_sqrt_22_batch(self.sigma_running)
        # [B, C*H*W, 2, 1]
        z_centered = (z - self.mu_running).reshape(B * C * H * W, 2).unsqueeze(-1)
        o = torch.bmm(
            sigma_inv_sqrt.tile(B, 1, 1).reshape(B * C * H * W, 2, 2), z_centered
        )
        return o

    def forward(self, z: torch.Tensor):
        # z : [B, C, H, W] (complex)

        B, C, H, W = z.shape
        if not (self.running_avg_initilized):
            self.init_running_avg(C * H * W)

        z = torch.view_as_real(z)

        if self.training:
            self.update_mu_sigma(z)
            # [B*C*H*W, 2]
            o = self.normalize(z)
        else:
            # [B*C*H*W, 2]
            o = self.normalize_inference(z)

        if self.affine:
            # [B*C*H*W, 2, 1]
            o = torch.bmm(self.gamma.tile(B * C * H * W, 1, 1), o) + self.beta.reshape(
                1, 2, 1
            )

        # [B, C, H, W, 2]
        o = o.view(B, C, H, W, 2)
        return torch.view_as_complex(o)
