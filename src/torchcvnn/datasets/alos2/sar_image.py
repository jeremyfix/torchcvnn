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

# Format described p84 of
# https://www.eorc.jaxa.jp/ALOS/en/alos-2/pdf/product_format_description/PALSAR-2_xx_Format_CEOS_E_g.pdf
descriptor_format = [
    ("format_control_document_id", 16, 12, "A", "CEOS-SAR    "),
    ("file_id", 48, 16, "A", None),
    ("number_data_records", 180, 6, "I", None),
    ("number_bytes_prefix_data", 276, 4, "I", None),  # 544 for L1.1, 192 for L1.5/3.1
    ("number_bytes_data_record", 280, 8, "I", None),
    ("number_bytes_suffix_data", 288, 4, "I", None),  # 0
    ("sample_data_number_locator", 296, 8, "A", None),
    ("sar_data_format_type", 428, 4, "A", None)  # e.g L1.1 'C*8b'
    # ("scanscar_burst_information", 456, 4, "I", None),
]
descriptor_record_length = 720


data_records_format = [
    ("record_type_code", 5, 1, "B", 10),  # To check we are aligned with the file format
    ("count_data_pixels", 24, 4, "B", None),
    ("transmitted_pulse_polarization", 52, 2, "B", None),  # either H(0) or V(1)
    ("received_pulse_polarization", 54, 2, "B", None),  # either H(0) or V(1)
    ("chirp_length", 68, 4, "B", None),
    ("chirp_constant_coefficient", 72, 4, "B", None),
    ("chirp_linear_coefficient", 76, 4, "B", None),
    ("chirp_quadratic_coefficient", 80, 4, "B", None),
    ("receiver_gain", 92, 4, "B", None),
    ("slant_range,_first_data", 116, 4, "B", None),
    ("latitude_first", 192, 4, "B", None),  # 1/1,000,000 deg
    ("latitude_last", 200, 4, "B", None),  # 1/1,000,000 deg
    ("longitude_first", 204, 4, "B", None),  # 1/1,000,000 deg
    ("longitude_last", 212, 4, "B", None),  # 1/1,000,000 deg
]


def parse_image_data(fh, number_pixels, number_records):
    data_bytes = fh.read(number_records * number_pixels * 8)
    datas = struct.unpack(">" + ("i" * number_records * number_pixels * 2), data_bytes)
    cplx_datas = [real + 1j * imag for (real, imag) in zip(datas[0::2], datas[1::2])]
    datas = np.array(cplx_datas).reshape(number_records, number_pixels)

    return datas


class SARImage:
    r"""
    Processing of the SAR Image
    """

    def __init__(self, filepath):
        self.descriptor_records = {}
        self.data_records = {}
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
            parse_utils.parse_from_format(
                fh,
                self.data_records,
                data_records_format,
                1,
                0,  # Note: this is variable, don't trust the fh_offset from this call
                fh_offset,
            )

            # Rewind the head to the beginning of the data records
            fh.seek(fh_offset + self.descriptor_records["number_bytes_prefix_data"])
            number_records = self.descriptor_records["number_data_records"]
            number_pixels = self.data_records["count_data_pixels"]
            self.data = parse_image_data(fh, number_pixels, number_records)

    def __repr__(self):
        descriptor_txt = parse_utils.format_dictionary(self.descriptor_records, 1)
        data_txt = parse_utils.format_dictionary(self.data_records, 1)
        return f"""
Descriptor : 
{descriptor_txt}

Data:
{data_txt}
        """
