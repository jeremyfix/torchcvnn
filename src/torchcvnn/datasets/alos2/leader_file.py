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
import numpy as np

# Local imports
from . import parse_utils

descriptor_format = [
    ("file_id", 48, 16, "A", None),
    ("number_map_projection_record", 192, 6, "I", None),
]
descriptor_record_length = 720

summary_format = [
    ("scene_id", 20, 32, "A", None),
    ("number_sar_channels", 388, 4, "I", None),
    ("sensor_id", 412, 32, "A", None),
    ("line_content_indicator", 1670, 8, "A", None),  # RANGEbbb for L1.1 or OTHERbbb
    ("line_spacing", 1686, 16, "F16.7", None),
    ("pixel_spacing", 1702, 16, "F16.7", None),
    ("doppler_center_frequency_constant_term", 1734, 16, "F16.7", None),
    ("doppler_center_frequency_linear_term", 1750, 16, "F16.7", None),
    ("nominal_off_nadir", 1838, 16, "F16.7", None),
]
summary_record_length = 4096

map_projection_format = []
map_projection_record_length = 1620

platform_projection_format = []
platform_projection_record_length = 4680

altitude_data_format = [
    ("pitch", 40, 14, "E14.6", None),
    ("roll", 54, 14, "E14.6", None),
    ("yaw", 68, 14, "E14.6", None),
]
altitude_data_record_length = 16384

radiometric_data_format = [
    ("record_number", 0, 4, "B", None),
    ("calibration_factor", 20, 16, "F16.7", None),
]
radiometric_data_record_length = 9860

data_quality_format = []
data_quality_record_length = 1620

facility_format = []
facility_record_length = 325000 + 511000 + 3072 + 728000 + 5000


class LeaderFile:
    r"""
    Processing of the SAR trailer file.
    """

    def __init__(self, filepath):
        self.descriptor_record = {}
        self.summary_record = {}
        self.altitude_record = {}
        self.radiometric_record = {}
        with open(filepath, "rb") as fh:
            fh_offset = 0
            fh_offset = parse_utils.parse_from_format(
                fh,
                self.descriptor_record,
                descriptor_format,
                1,
                descriptor_record_length,
                fh_offset,
            )
            fh_offset = parse_utils.parse_from_format(
                fh,
                self.summary_record,
                summary_format,
                1,
                summary_record_length,
                fh_offset,
            )

            # Skip map projection record
            # Does not exist in L1.1
            if self.descriptor_record["number_map_projection_record"] == 1:
                fh_offset += map_projection_record_length

            # Skip platform projection record
            fh_offset += platform_projection_record_length

            # Altitude data record
            fh_offset = parse_utils.parse_from_format(
                fh,
                self.altitude_record,
                altitude_data_format,
                1,
                altitude_data_record_length,
                fh_offset,
            )

            # Radiometric data record
            fh_offset = parse_utils.parse_from_format(
                fh,
                self.radiometric_record,
                radiometric_data_format,
                1,
                radiometric_data_record_length,
                fh_offset,
            )

    @property
    def calibration_factor(self):
        cf = self.radiometric_record["calibration_factor"]
        return np.sqrt(10.0 ** ((cf - 32.0) / 10.0))

    def __repr__(self):
        descriptor_txt = parse_utils.format_dictionary(self.descriptor_record, 1)
        summary_txt = parse_utils.format_dictionary(self.summary_record, 1)
        altitude_txt = parse_utils.format_dictionary(self.altitude_record, 1)
        radiometric_txt = parse_utils.format_dictionary(self.radiometric_record, 1)
        return f"""
Descriptor:
{descriptor_txt}
Summary:
{summary_txt}
Altitude:
{altitude_txt}
Radiometric:
{radiometric_txt}
        """
