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
from typing import Union
import pathlib

# Local imports
from . import parse_utils

descriptor_format = [
    ("record_sequence_number", 0, 4, "B", 1),
    ("first_record_subtype_code", 4, 1, "B", 192),
    ("record_type_code", 5, 1, "B", 192),
    ("second_subtype_code", 6, 1, "B", 18),
    ("third_subtype_code", 7, 1, "B", 18),
    ("length_record", 8, 4, "B", 360),
    ("flag", 12, 2, "A", "A "),
    # ("blanks", 14, 2, "A", "  "),
    ("superstructure_doc_id", 16, 12, "A", "CEOS-SAR    "),
    ("superstructure_doc_rev_level", 28, 2, "A", " A"),
    ("superstructure_fmt_rev_level", 30, 2, "A", " A"),
    ("software_release_level", 32, 12, "A", None),
    ("physical_volume_id", 44, 16, "A", None),
    ("logical_volume_id", 60, 16, "A", None),
    ("volume_set_id", 76, 16, "A", None),
    ("total_number_volumes", 92, 2, "I", 1),
    ("physical_volume_seq_num_first", 94, 2, "I", 1),
    ("physical_volume_seq_num_last", 96, 2, "I", 1),
    ("physical_volume_seq_num_cur", 98, 2, "I", 1),
    ("file_number", 100, 4, "I", None),
    ("logical_volume_within_volume", 104, 4, "I", None),
    ("logical_volume_within_physical", 108, 4, "I", None),
    ("logical_volume_creation_date", 112, 8, "A", None),  # YYYYMMDD
    ("logical_volume_creation_time", 120, 8, "A", None),  # HHMMSSXX
    ("logical_volume_creation_country", 128, 12, "A", None),
    ("logical_volume_creation_agency", 140, 8, "A", None),
    ("logical_volume_generation_facility", 148, 12, "A", None),
    ("number_of_file_pointer_records", 160, 4, "I", None),
    ("number_of_text_records", 164, 4, "I", None),
    # Volume descriptor spare A92 + local use segment A100
]
volume_descriptor_record_length = 360

file_pointer_format = [
    ("record_number", 0, 4, "B", None),
    ("reference_id", 20, 16, "A", None),
    ("reference_file_class_code", 64, 4, "A", None),
]
file_pointer_record_length = 360

text_records_format = [
    ("record_number", 0, 4, "B", None),
    ("product_id", 16, 40, "A", None),
    ("scene_id", 156, 40, "A", None),
    ("scene_location_id", 196, 40, "A", None),
]
text_record_length = 360


class VolFile:
    r"""
    Processing a Volume Directory file in the CEOS format. The parsed
    informations can be accessed through the attributes `descriptor_records`,
    `file_pointer_records` and `text_records`

    Arguments:
        filepath: the path to the volume directory file
    """

    def __init__(self, filepath: Union[str, pathlib.Path]):
        self.descriptor_records = {}
        self.file_pointer_records = []
        self.text_record = {}

        with open(filepath, "rb") as fh:
            # Parsing the volume descriptor
            fh_offset = 0
            fh_offset = parse_utils.parse_from_format(
                fh,
                self.descriptor_records,
                descriptor_format,
                1,
                volume_descriptor_record_length,
                fh_offset,
            )

            # Parsing the file pointer
            number_of_file_pointer_records = self.descriptor_records[
                "number_of_file_pointer_records"
            ]
            fh_offset = parse_utils.parse_from_format(
                fh,
                self.file_pointer_records,
                file_pointer_format,
                number_of_file_pointer_records,
                file_pointer_record_length,
                fh_offset,
            )

            # Parsing the file pointer
            fh_offset = parse_utils.parse_from_format(
                fh,
                self.text_record,
                text_records_format,
                1,
                text_record_length,
                fh_offset,
            )

    def __repr__(self):
        descriptor_txt = parse_utils.format_dictionary(self.descriptor_records, 1)
        # text_txt = parse_utils.format_dictionary(self.text_records, 1)
        fp_texts = ""
        for i, fi in enumerate(self.file_pointer_records):
            fp_texts += f"File pointer {i} : \n" + parse_utils.format_dictionary(fi, 2)
            fp_texts += "\n"

        text_texts = parse_utils.format_dictionary(self.text_record, 1)
        return f"""
Descriptor:
{descriptor_txt}
File pointers : {len(self.file_pointer_records)} records
{fp_texts}

Text records:
{text_texts}
        """
