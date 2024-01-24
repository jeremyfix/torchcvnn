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
# A SAR Image file contains
# - A File descriptor with 720 bytes
# - For L1.1 : Signal data

descriptor_format = [
    ("format_control_document_id", 16, 12, "A", "CEOS-SAR    "),
    ("file_id", 48, 16, "A", None),
    ("number_data_records", 180, 6, "I", None),
    ("sar_data_record_length", 186, 6, "I", None),
    ("bit_length_per_sample", 216, 4, "I", None),
    ("num_samples_per_data_group", 220, 4, "I", None),
    ("num_bytes_per_data_group", 224, 4, "I", None),
    ("number_bytes_prefix_data", 276, 4, "I", None),  # 544 for L1.1, 192 for L1.5/3.1
    ("number_bytes_data_record", 280, 8, "I", None),
    ("number_bytes_suffix_data", 288, 4, "I", None),  # 0
    ("sample_data_number_locator", 296, 8, "A", None),
    ("sar_data_format_type", 428, 4, "A", None),  # e.g L1.1 'C*8b'
    ("scanscar_num_burst", 448, 4, "I", None),
    ("scanscar_num_lines_per_burst", 452, 4, "I", None),
    ("scanscar_burst_information", 456, 4, "I", None),
]
descriptor_record_length = 720


data_records_format = [
    ("record_sequence_number", 0, 4, "B", None),
    ("record_type_code", 5, 1, "B", 10),  # To check we are aligned with the file format
    ("record_length", 8, 4, "B", None),
    ("sar_image_data_line_number", 12, 4, "B", None),
    ("num_left_fill_pixels", 20, 4, "B", 0),  # No left fill in 1.1, hence 0 is expected
    ("count_data_pixels", 24, 4, "B", None),
    (
        "num_right_fill_pixels",
        28,
        4,
        "B",
        0,
    ),  # No right fill in 1.1, hence 0 is expected
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
    ("ALOS2_frame_number", 284, 4, "B", 0),
    ("auxiliary_data", 288, 256, "B", 0),
]
data_record_header_length = 544


def parse_image_data(fh, base_offset, number_records):
    # For some reasons, the following which I expect to be faster
    # is taking much more memory than the line by line approach below
    # print("Parsing image data")
    # data_bytes = fh.read(number_records * number_pixels * 8)
    # datas = struct.unpack(">" + ("i" * number_records * number_pixels * 2), data_bytes)
    # cplx_datas = [real + 1j * imag for (real, imag) in zip(datas[0::2], datas[1::2])]
    # del datas

    # print("Cplx data")
    # datas = np.array(cplx_datas).reshape(number_records, number_pixels) / 2**16
    # del cplx_datas
    # print("As numpy")

    lines = []
    record_info = {}
    for i in range(number_records):
        # print(base_offset)
        # Read the header and shift the file pointer
        base_offset = parse_utils.parse_from_format(
            fh,
            record_info,
            data_records_format,
            1,
            data_record_header_length,
            base_offset,
        )
        assert i == (record_info["sar_image_data_line_number"] - 1)

        number_pixels = record_info["count_data_pixels"]

        fh.seek(base_offset)
        data_bytes = fh.read(number_pixels * 8)
        # TODO: is that the correct unpacking format ?!
        datas = struct.unpack(">" + ("f" * (number_pixels * 2)), data_bytes)

        cplx_datas = [real + 1j * imag for (real, imag) in zip(datas[::2], datas[1::2])]
        cplx_datas = np.array(cplx_datas)

        lines.append(cplx_datas)

        # Shift the base_offset to the next record header beginning
        base_offset += 8 * number_pixels

    datas = np.array(lines)

    return datas


class SARImage:
    r"""
    Processing of the SAR Image
    """

    def __init__(self, filepath):
        self.descriptor_records = {}
        self.data_records = {}
        self.filepath = filepath

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

            # Read the header of the first record
            fh_offset = parse_utils.parse_from_format(
                fh,
                self.data_records,
                data_records_format,
                1,
                data_record_header_length,
                fh_offset,
            )

    @property
    def num_rows(self):
        return self.descriptor_records["number_data_records"]

    @property
    def num_cols(self):
        return self.data_records["count_data_pixels"]

    def read_patch(self, start_line, num_lines, start_col, num_cols):
        number_of_pixels_per_line = self.data_records["count_data_pixels"]
        base_offset = (
            descriptor_record_length
            + (data_record_header_length + 8 * number_of_pixels_per_line) * start_line
        )
        lines = []
        with open(self.filepath, "rb") as fh:
            for i in range(num_lines):
                offset = base_offset + i * (
                    data_record_header_length + 8 * number_of_pixels_per_line
                )

                # Move the file descriptor to the beginning of the record
                fh.seek(offset)

                # And read/decode the record header
                record = {}
                parse_utils.parse_from_format(
                    fh,
                    record,
                    data_records_format,
                    1,
                    data_record_header_length,
                    offset,
                )

                # Move the fh up to the columns to read
                fh.seek(8 * start_col, 1)  # whence=1 for relative shift
                data_bytes = fh.read(num_cols * 8)
                datas = struct.unpack(">" + ("f" * (num_cols * 2)), data_bytes)
                cplx_datas = [
                    real + 1j * imag for (real, imag) in zip(datas[::2], datas[1::2])
                ]
                cplx_datas = np.array(cplx_datas)
                lines.append(cplx_datas)
        return np.array(lines)

    def __repr__(self):
        descriptor_txt = parse_utils.format_dictionary(self.descriptor_records, 1)
        data_txt = parse_utils.format_dictionary(self.data_records, 1)
        return f"""
Descriptor : 
{descriptor_txt}

Data:
{data_txt}
        """
