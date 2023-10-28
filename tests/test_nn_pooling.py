# External imports
import torch

# Local imports
import torchcvnn
import torchcvnn.nn as nn_c


def test_mpool2d():
    mpool = nn_c.MaxPool2d(kernel_size=2, stride=2, padding=0)

    N = 10
    inputs = torch.tensor([[i + j * 1j for i in range(N)] for j in range(N)])
    inputs = inputs.unsqueeze(0).unsqueeze(0)
    outputs = mpool(inputs)

    expected_outputs = torch.tensor(
        [[2 * i + 1 + (2 * j + 1) * 1j for i in range(N // 2)] for j in range(N // 2)]
    )
    assert torch.allclose(outputs, expected_outputs)


if __name__ == "__main__":
    test_mpool2d()
