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
import logging
import struct

# External imports
import torch
from torch.utils.data import Dataset
import numpy as np


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

    return parsed_fields


class MSTARSample:

    def __init__(self, filename: str):
        self.filename = pathlib.Path(filename)

        with open(self.filename, "r", errors="replace") as fh:
            # Read the header from the file
            self.header = parse_header(fh)

            num_rows = int(self.header["NumberOfRows"])
            num_cols = int(self.header["NumberOfColumns"])
            phoenix_header_length = int(self.header["PhoenixHeaderLength"])
            native_header_length = int(self.header["native_header_length"])

            fh.seek(phoenix_header_length + native_header_length)

            sig_size = int(self.header["PhoenixSigSize"])
            bytes_per_values = (
                sig_size - phoenix_header_length - native_header_length
            ) // (2 * num_rows * num_cols)

            if bytes_per_values == 4:
                # Read the data as float32
                pass
            elif bytes_per_values == 2:
                # Read the data as uint16
                pass
            else:
                raise ValueError(
                    f"Unsupported number of bytes per value : {bytes_per_values}"
                )

        # Read the data from the file
        with open(self.filename, "rb") as fh:
            fh.seek(phoenix_header_length + native_header_length)

            data_bytes = fh.read(num_rows * num_cols * bytes_per_values)
            unpacked = struct.unpack(">" + ("f" * num_cols * num_rows), data_bytes)
            magnitudes = np.array(unpacked).reshape(num_rows, num_cols)

            data_bytes = fh.read(num_rows * num_cols * bytes_per_values)
            unpacked = struct.unpack(">" + ("f" * num_cols * num_rows), data_bytes)
            phases = np.array(unpacked).reshape(num_rows, num_cols)

            self.data = magnitudes * np.exp(1j * phases)


class MSTARTargets(Dataset):

    def __init__(self, rootdir: str, train: bool, transform=None):
        self.rootdir = pathlib.Path(rootdir)
        self.transform = transform

        self.class_names = ["BMP2", "BTR70", "T82", "SLICY"]
        self.data_files = {}
        self.num_data_files = {}
        self.tot_num_data_files = 0
        if train:
            for key in self.class_names:
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
        logging.debug(f"Loading the MSTAR file {filename}")

        sample = MSTARSample(filename)
        class_idx = self.class_names.index(key)

        data = sample.data
        if self.transform is not None:
            data = self.transform(data)

        return data, class_idx
