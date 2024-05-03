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
import pathlib

# Local imports
from .ann_file import AnnFile


def parse_slc_filename(filename):
    """
    Parses a filename of a SLC file
    {site name}_{line ID}_{flight ID}_{data take counter}_{acquisition date}_{band}{steering}{polarization}_{stack_version}... _{baseline correction}_{segment number}_{downsample factor}.slc

    and returns all the information in a dictionary
    """
    # Remove the .slc extension and split the fields
    fields = filename[:-4].split("_")
    parameters = {
        "site_name": fields[0],
        "line_ID": fields[1],
        "flight_ID": fields[2],
        "data_take_counter": fields[3],
        "acquisition_date": fields[4],
        "band": fields[5][0],
        "steering": fields[5][1:-2],
        "polarization": fields[5][-2:],
        "stack_version": fields[6],
        "baseline_correction": fields[7],
        "segment_number": int(
            fields[8][1:]
        ),  # the segment is encoded as  s{segment_number}
        "downsample_factor": fields[9],
    }
    return parameters


class SLCFile:
    r"""
    Reads a SLC file

    The filenames contain interesting information:

    {site name}_{line ID}_{flight ID}_{data take counter}_{acquisition date}_{stack number}_{band}{steering}{polarization}_{stack_version}... _{baseline correction}_{segment number}_{downsample factor}.slc

    e.g. SSurge_15305_14170_007_141120_L090HH_01_BC_s1_1x1.slc is

    - site_name : SSurge
    - line ID : 15305
    - flight ID : 14170
    - data take counter : 007
    - acquisition date : 141120, the date is in YYMMDD format (UTC time).
    - stack number : L090
    - band : HH
    - steering : 01
    - polarization : HH
    - stack version : 01
    - baseline correction : BC, means the data is corrected for residual baseline
    - segment number : s1
    - downsample factor : 1x1

    There is one SLC file per segment and per polarization.
    """

    def __init__(self, filename):
        self.filename = pathlib.Path(filename)
        self.parameters = parse_slc_filename(self.filename.name)

        # The annotation filename is almost the same as the SLC filename, except we drop
        # the segment number and downsample factor
        # We expect it to be colocated with the SLC file
        ann_filename = "_".join(str(self.filename.name).split("_")[:-2]) + ".ann"
        self.ann_file = AnnFile(str(self.filename.parent / ann_filename))

        downsample_factor = self.parameters["downsample_factor"]
        segment_number = self.parameters["segment_number"]
        self.azimuth_pixel_spacing = getattr(
            self.ann_file, f"{downsample_factor}_slc_azimuth_pixel_spacing"
        )
        self.range_pixel_spacing = getattr(
            self.ann_file, f"{downsample_factor}_slc_range_pixel_spacing"
        )
        self.global_average_squint_angle = self.ann_file.global_average_squint_angle
        self.center_wavelength = self.ann_file.center_wavelength
        self.n_rows = getattr(
            self.ann_file, f"slc_{segment_number}_{downsample_factor}_rows"
        )
        self.n_cols = getattr(
            self.ann_file, f"slc_{segment_number}_{downsample_factor}_columns"
        )
