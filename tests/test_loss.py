# MIT License

# Copyright (c) 2024 Quentin Gabot

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
import numpy as np

# Local imports
import torchcvnn.nn as c_nn


def test_complex_mse_loss():
    y_true = torch.empty(1, 3, dtype=torch.complex64)
    y_pred = torch.empty(1, 3, dtype=torch.complex64)
    loss_mse = c_nn.ComplexMeanSquareError()
    loss_mse(y_true, y_pred)


def test_complex_vae_loss():
    y_true = torch.empty(1, 3, dtype=torch.complex64)
    y_pred = torch.empty(1, 3, dtype=torch.complex64)
    mu = torch.empty(1, 2, dtype=torch.complex64)
    logvar = torch.empty(1, 2, dtype=torch.complex64)
    loss_vae = c_nn.ComplexVAELoss()
    loss_vae(y_true, y_pred, mu, logvar)


if __name__ == "__main__":
    test_complex_mse_loss()
    test_complex_vae_loss()
