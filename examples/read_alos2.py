from pathlib import Path
from torchcvnn.datasets import alos2

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


def test1():
    rootdir = Path("/mounts/Datasets1/Polarimetric-SanFrancisco/SAN_FRANCISCO_ALOS2/")

    print("===== Volume =====")
    vol_filepath = rootdir / "VOL-ALOS2044980750-150324-HBQR1.1__A"
    volFile = alos2.VolFile(vol_filepath)
    print(volFile.descriptor_records)
    print(volFile.file_pointer_records)
    print(volFile.text_records)

    print("===== Trailer =====")
    trailer_filepath = rootdir / "TRL-ALOS2044980750-150324-HBQR1.1__A"
    trailFile = alos2.TrailerFile(trailer_filepath)
    print(trailFile)


if __name__ == "__main__":
    test1()
