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
from typing import Tuple
from glob import glob
import pathlib

# External imports
import torch
from torch.utils.data import Dataset


def parse_header(fh):
    parsed_fields = {}
    _ = fh.readline()
    start_line = fh.readline().strip()
    if start_line != "[PhoenixHeaderVer01.04]":
        raise ValueError(
            f"Invalid header : {start_line}, expected [PhoenixHeaderVer01.04]"
        )
    next_line = fh.readline().strip()
    while next_line != "[EndofPhoenixHeader]":
        items = next_line.split("=")
        key = items[0].strip()
        value = items[1].strip()
        parsed_fields[key] = value
        next_line = fh.readline().strip()
    print(next_line + "ENNND")

    return parsed_fields


class MSTARSample:

    def __init__(self, filename: str):
        self.filename = pathlib.Path(filename)

        with open(self.filename, "r", errors="replace") as fh:
            # Read the header from the file
            self.header = parse_header(fh)

            num_rows = int(self.header["NumberOfRows"])
            num_cols = int(self.header["NumberOfColumns"])

            # Read the data from the file


class MSTARTargets(Dataset):

    def __init__(self, rootdir: str, train: bool, transform=None):
        self.rootdir = pathlib.Path(rootdir)
        self.transform = transform

        self.data_files = {}
        self.num_data_files = {}
        self.tot_num_data_files = 0
        if train:
            for key in ["BMP2", "BTR70", "T82"]:
                pattern = str(
                    self.rootdir / "TARGETS" / "TRAIN" / "17_DEG" / key / "**" / "HB*"
                )
                self.data_files[key] = glob(pattern, recursive=True)
                self.num_data_files[key] = len(self.data_files[key])
                self.tot_num_data_files += self.num_data_files[key]

    def __len__(self) -> int:
        return self.tot_num_data_files

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        if index >= self.tot_num_data_files:
            raise IndexError

        for key in self.data_files.keys():
            if index < self.num_data_files[key]:
                break
            index -= self.num_data_files[key]

        filename = self.data_files[key][index]
        sample = MSTARSample(filename)
        print(f"Loading {filename}...")
        return None
