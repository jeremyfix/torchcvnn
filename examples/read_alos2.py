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


def read_field(fh, start_byte, num_bytes, type_bytes):
    fh.seek(start_byte)
    data_bytes = fh.read(num_bytes)
    if type_bytes == "A":
        return data_bytes.decode("ascii")
    elif type_bytes == "B":
        # Binary number representation, big_endian
        return int.from_bytes(data_bytes, "big")


def test1():
    rootdir = Path("/mounts/Datasets1/Polarimetric-SanFrancisco/SAN_FRANCISCO_ALOS2/")

    vol_file = rootdir / "VOL-ALOS2044980750-150324-HBQR1.1__A"
    with open(vol_file, "rb") as fh:
        record_sequence_number = read_field(fh, 0, 4, "B")
        assert record_sequence_number == 1

        fst_record_subtype = read_field(fh, 4, 1, "B")
        assert fst_record_subtype == 192

        record_type_code = read_field(fh, 5, 1, "B")
        assert record_type_code == 192

        flag = read_field(fh, 12, 2, "A")
        assert flag == "A "

        superstructure = read_field(fh, 30, 2, "A")
        assert superstructure == " A"

        logical_volume = read_field(fh, 60, 16, "A")


if __name__ == "__main__":
    test1()
