# MIT License

# Copyright (c) 2024 Quentin Gabot

# The following is an adaptation of the initialization code from
# PyTorch library for the complex-valued neural networks

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

import torch
import torch.nn as nn


class ComplexMeanSquareError(nn.Module):
    def __init__(self):
        super(ComplexMeanSquareError, self).__init__()

    def forward(self, y_true, y_pred):
        # Ensure y_pred and y_true are torch tensors
        y_pred = torch.as_tensor(y_pred)
        y_true = torch.as_tensor(y_true)

        # Handle complex pred but real true
        if torch.is_complex(y_pred) and not torch.is_complex(y_true):
            y_true = torch.complex(y_true, torch.zeros_like(y_true))

        # Ensure y_true is the same dtype as y_pred
        y_true = y_true.type_as(y_pred)

        # Calculate Mean Square Error
        mse = torch.mean(torch.square(torch.abs(y_true - y_pred)))

        return mse


class ComplexVAELoss(nn.Module):
    def __init__(self):
        """
        Initializes the VAE Loss module.

        """
        super(ComplexVAELoss, self).__init__()

    def forward(self, x, recon_x, mu, logvar):
        """
        Computes the VAE loss.

        Parameters:
        recon_x: Reconstructed data.
        x: Original input data.
        mu: Mean from the latent space.
        logvar: Log variance from the latent space.

        Returns:
        torch.Tensor: Computed VAE loss.
        """
        # Reconstruction Loss
        MSELoss = ComplexMeanSquareError()
        recon_loss = MSELoss(y_true=x, y_pred=recon_x)

        # KL Divergence
        kl_divergence = torch.abs(
            -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        )  # valeur absolue car la loss doit être réelle ?
        return recon_loss + kl_divergence
