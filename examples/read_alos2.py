from pathlib import Path
from torchcvnn.datasets import alos2
import matplotlib

# matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import skimage
from skimage import exposure

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


def normalize(data):
    mod = np.abs(data)
    mod = 20 * np.log10(mod + 1e-10)

    mod = (mod - mod.min()) / (mod.max() - mod.min())  # rescale between 0 and 1
    p2, p98 = np.percentile(mod, (2, 98))

    mod = exposure.rescale_intensity(mod, in_range=(p2, p98))
    mod = skimage.img_as_float(mod)
    return mod


def test1():
    rootdir = Path("/home/fix_jer/Tools/SARData/SAN_FRANCISCO_ALOS2")
    # rootdir = Path("/mounts/Datasets1/Polarimetric-SanFrancisco/SAN_FRANCISCO_ALOS2/")

    print("===== Volume =====")
    vol_filepath = rootdir / "VOL-ALOS2044980750-150324-HBQR1.1__A"
    volFile = alos2.VolFile(vol_filepath)
    print(volFile)

    print("===== Trailer =====")
    trailer_filepath = rootdir / "TRL-ALOS2044980750-150324-HBQR1.1__A"
    trailFile = alos2.TrailerFile(trailer_filepath)
    print(trailFile)

    print("===== Leader =====")
    leader_filepath = rootdir / "LED-ALOS2044980750-150324-HBQR1.1__A"
    leaderFile = alos2.LeaderFile(leader_filepath)
    print(leaderFile)

    print("===== SAR HH Image =====")
    hh_filepath = rootdir / "IMG-HH-ALOS2044980750-150324-HBQR1.1__A"
    HH_Image = alos2.SARImage(hh_filepath, num_max_records=5)
    print(HH_Image.data[0, :5])

    print("===== SAR HV Image =====")
    hv_filepath = rootdir / "IMG-HV-ALOS2044980750-150324-HBQR1.1__A"
    HV_Image = alos2.SARImage(hv_filepath, num_max_records=5)
    print(HV_Image.data[0, :5])

    print("===== SAR VV Image =====")
    vv_filepath = rootdir / "IMG-VV-ALOS2044980750-150324-HBQR1.1__A"
    VV_Image = alos2.SARImage(vv_filepath, num_max_records=1000)
    print(VV_Image.data[0, :5])

    mod = normalize(VV_Image.data)

    plt.figure()
    plt.imshow(mod, cmap="gray")
    # plt.savefig("HH.png", bbox_inches="tight")
    # plt.close()
    plt.show()


if __name__ == "__main__":
    test1()
