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

# External imports
import torch

# Local imports
import torchcvnn.nn as c_nn


def test_transformer_encoder_layer():
    nhead = 8
    seq_len = 10
    batch_size = 32
    num_features = 512

    encoder_layer = c_nn.TransformerEncoderLayer(d_model=num_features, nhead=nhead)
    src = torch.rand(seq_len, batch_size, num_features)
    out = encoder_layer(src)

    assert out.shape(seq_len, batch_size, num_features)


def test_transformer_encoder():
    nhead = 8
    seq_len = 10
    batch_size = 32
    num_features = 512

    encoder_layer = c_nn.TransformerEncoderLayer(d_model=num_features, nhead=nhead)
    transformer_encoder = c_nn.TransformerEncoder(encoder_layer, num_layers=4)

    src = torch.randn((seq_len, batch_size, num_features), dtype=torch.complex64)
    out = transformer_encoder(src)
    assert out.shape == (
        seq_len,
        batch_size,
    )


if __name__ == "__main__":
    test_transformer_encoder_layer()
    test_transformer_encoder()
