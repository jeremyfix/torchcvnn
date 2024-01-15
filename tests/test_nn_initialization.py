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

# Local imports
import torchcvnn.nn as c_nn


def test_complex_kaiming_normal():
    w = torch.empty(3, 5, dtype=torch.complex64)
    c_nn.init.complex_kaiming_normal_(w, mode="fan_in", nonlinearity="relu")


def test_complex_kaiming_uniform_():
    w = torch.empty(3, 5, dtype=torch.complex64)
    c_nn.init.complex_kaiming_uniform_(w, mode="fan_in", nonlinearity="relu")


def test_complex_xavier_uniform_():
    w = torch.empty(3, 5, dtype=torch.complex64)
    c_nn.init.complex_xavier_uniform_(w, nonlinearity="relu")


def test_complex_xavier_normal_():
    w = torch.empty(3, 5, dtype=torch.complex64)
    c_nn.init.complex_xavier_normal_(w, nonlinearity="relu")


if __name__ == "__main__":
    test_complex_kaiming_normal()
    test_complex_kaiming_uniform_()
    test_complex_xavier_uniform_()
    test_complex_xavier_normal_()
