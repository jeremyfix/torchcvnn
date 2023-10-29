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
import torchcvnn.nn as nn_c


def test_mpool2d():
    mpool = nn_c.MaxPool2d(kernel_size=2, stride=2, padding=0)

    N = 10
    inputs = torch.tensor([[i + j * 1j for i in range(N)] for j in range(N)])
    inputs = inputs.unsqueeze(0).unsqueeze(0)
    outputs = mpool(inputs)

    expected_outputs = (
        torch.tensor(
            [
                [2 * i + 1 + (2 * j + 1) * 1j for i in range(N // 2)]
                for j in range(N // 2)
            ]
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )
    assert torch.allclose(outputs, expected_outputs)


def test_avgpool2d():
    avgpool = nn_c.AvgPool2d(kernel_size=3, stride=1, padding=1)

    N = 10
    inputs = torch.tensor([[i + j * 1j for i in range(N)] for j in range(N)])
    inputs = inputs.unsqueeze(0).unsqueeze(0)
    outputs = avgpool(inputs)

    # With kernel_size=3, stride=1, padding=1, output is the same shape
    # as the input
    assert inputs.shape == outputs.shape

    def num_series(row_or_col):
        return 3 if row_or_col > 0 and row_or_col < N - 1 else 2

    expected_outputs = (
        torch.tensor(
            [
                [
                    (num_series(j) * (i - 1 + i + i + 1)) / 9.0 * 1j
                    + (num_series(i) * (j - 1 + j + j + 1)) / 9.0
                    for j in range(N)
                ]
                for i in range(N)
            ]
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )
    # Handle the border cases
    expected_outputs[0, 0, 1:-1, 0] += 3.0 / 9.0
    expected_outputs[0, 0, 0, 0] += 2.0 / 9.0
    expected_outputs[0, 0, -1, 0] += 2.0 / 9.0
    expected_outputs[0, 0, 1:-1, -1] -= 3.0 * N / 9.0
    expected_outputs[0, 0, 0, -1] -= 2.0 * N / 9.0
    expected_outputs[0, 0, -1, -1] -= 2.0 * N / 9.0

    expected_outputs[0, 0, 0, 1:-1] += 3.0 / 9.0 * 1j
    expected_outputs[0, 0, 0, 0] += 2.0 / 9.0 * 1j
    expected_outputs[0, 0, 0, -1] += 2.0 / 9.0 * 1j
    expected_outputs[0, 0, -1, 1:-1] -= 3.0 * N / 9.0 * 1j
    expected_outputs[0, 0, -1, 0] -= 2.0 * N / 9.0 * 1j
    expected_outputs[0, 0, -1, -1] -= 2.0 * N / 9.0 * 1j

    assert torch.allclose(outputs, expected_outputs)


if __name__ == "__main__":
    test_mpool2d()
    test_avgpool2d()
