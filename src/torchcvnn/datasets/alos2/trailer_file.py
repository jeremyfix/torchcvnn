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
import struct

# External imports
import numpy as np

# Local imports
from . import parse_utils

descriptor_format = [
    ("file_id", 48, 16, "A", None),
    ("number_low_resolution_image_records", 490, 6, "I", None),
    ("number_of_pixels", 504, 6, "I", None),
    ("number_of_lines", 510, 6, "I", None),
    ("image_record_length", 496, 8, "I", None),
]
descriptor_record_length = 720


def parse_image_data(fh, number_pixels, number_lines):
    data_bytes = fh.read(number_pixels * number_lines * 2)
    data_array = struct.unpack(">" + ("H" * number_pixels * number_lines), data_bytes)
    data = np.array(data_array, dtype=np.ushort).reshape(number_pixels, number_lines)
    return data


class TrailerFile:
    r"""
    Processing of the SAR trailer file.
    """

    def __init__(self, filepath):
        self.descriptor_records = {}
        with open(filepath, "rb") as fh:
            fh_offset = 0
            fh_offset = parse_utils.parse_from_format(
                fh,
                self.descriptor_records,
                descriptor_format,
                1,
                descriptor_record_length,
                fh_offset,
            )

            # Move the reading head to the beginning of the buffer
            fh.seek(fh_offset)
            num_pixels = self.descriptor_records["number_of_pixels"]
            num_lines = self.descriptor_records["number_of_lines"]
            self.image_data = parse_image_data(fh, num_pixels, num_lines)

    def __repr__(self):
        descriptor_txt = parse_utils.format_dictionary(self.descriptor_records, 1)
        return f"""
Descriptor : 
{descriptor_txt}
        """
