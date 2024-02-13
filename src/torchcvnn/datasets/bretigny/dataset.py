# MIT License

# Copyright (c) 2024 Chengfang Ren, Jeremy Fix

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
import pathlib
from typing import Tuple, Any

# External imports
from torch.utils.data import Dataset
import numpy as np


class Bretigny(Dataset):
    r"""
    Bretigny Dataset
    """

    """
    Class names
    """
    classes = ["0 - Unlabeld", "1 - Forest", "2 - Track", "3 - Urban", "4 - Fields"]

    def __init__(
        self,
        root: str,
        transform=None,
        balanced: bool = False,
        patch_size: tuple = (128, 128),
        patch_stride: tuple = None,
    ):
        self.root = pathlib.Path(root)

        sar_filename = self.root / "bretigny_seg.npz"
        if not sar_filename.exists():
            raise RuntimeError(f"Cannot find the file {sar_filename}")
        sar_data = np.load(sar_filename)
        self.HH, self.HV, self.VV = sar_data["HH"], sar_data["HV"], sar_data["VV"]

        if balanced:
            label_filename = self.root / "bretigny_seg_4ROI_balanced.npz"
        else:
            label_filename = self.root / "bretigny_seg_4ROI.npz"
        self.labels = np.load(label_filename)["arr_0"]

    def __len__(self) -> int:
        """
        Returns the total number of patches in the whole image.

        Returns:
            the total number of patches in the dataset
        """
        return 0

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        """
        Returns the indexes patch.

        Arguments:
            idx (int): Index

        Returns:
            tuple: (patch, labels) where patch contains the 4 complex valued polarization HH, HV, VH, VV and labels contains the aligned semantic labels
        """
        pass
