from pathlib import Path

"""
Example script to read ALOS-2 data for San Francisco Bay

The format is described in
https://www.eorc.jaxa.jp/ALOS/en/alos-2/pdf/product_format_description/PALSAR-2_xx_Format_CEOS_E_g.pdf

We have :
    IMG-{HH, HV, VH, VV}xxxxx__A
    LED-xxxx.1__A : SAR Leader file
    TRL-xxxx.1__A : SAR Trailer file
    VOL-xxxx.1__a : Volume directory file

    With ALOS2044980750-150324-HBQR1 as Sene ID - Product ID
    SceneID : ALOS2044980750-150324
        which stands for ALOS2 satelitte
                         04498 : orbit accumulation number
                         0750  : scene frame number
                         150324 : 2015/03/24
                            
    ProductID : HBQR1.1__A
        HBQ : High-sensitive mode Full (Quad.) polarimetry
          R : Right looking
          1.1 : Processing level
          _ : processing option not specified
          _ : map projection not specified
          A : Ascending orbit direction

In the format file, the San Francisco Data are described in Figure 3.1-5  p29

Volume Directory file: 
    Volume descriptor : 1 record of length 360 bytes
    File pointer : Number of pola x numnber of scans + 2; 360 bytes
    Text () : 1 record 360 bytes


"""


def read_field(fh, start_byte, num_bytes, type_bytes, expected):
    fh.seek(start_byte)
    data_bytes = fh.read(num_bytes)
    if type_bytes == "A":
        value = data_bytes.decode("ascii")
    elif type_bytes == "B":
        # Binary number representation, big_endian
        value = int.from_bytes(data_bytes, "big")
    elif type_bytes == "I":
        value = int(data_bytes.decode("ascii"))

    if expected is not None:
        assert value == expected

    return value


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


class VolFile:
    def __init__(self, filepath):
        self.descriptor_records = {}
        with open(filepath, "rb") as fh:
            parse_from_format(fh, self.descriptor_records, descriptor_format)


def parse_from_format(fh, obj, descriptor_format):
    for field_name, start_byte, num_bytes, type_bytes, expected in descriptor_format:
        value = read_field(fh, start_byte, num_bytes, type_bytes, expected)
        obj[field_name] = value


def test1():
    rootdir = Path("/mounts/Datasets1/Polarimetric-SanFrancisco/SAN_FRANCISCO_ALOS2/")

    vol_filepath = rootdir / "VOL-ALOS2044980750-150324-HBQR1.1__A"
    volFile = VolFile(vol_filepath)
    print(volFile.descriptor_records)


if __name__ == "__main__":
    test1()
