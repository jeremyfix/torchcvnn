# MIT License

# Copyright (c) 2024 Quentin Gabot

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
import os
import zipfile
import shutil
import pathlib

# External imports
import numpy as np
import requests
from tqdm import tqdm
import torch
import torch.utils.data
from torch.utils.data import Dataset
from PIL import Image

# Local imports
from .alos2 import ALOSDataset


class PolSFDataset(Dataset):
    r"""
    The Polarimetric SAR dataset provided by
    [https://ietr-lab.univ-rennes1.fr/polsarpro-bio/san-francisco/]()

    We expect the data to be already downloaded and available on your drive.

    Arguments:
        root: the top root dir where the data are downloaded and saved
        transform : the transform applied the cropped image
        patch_size: the dimensions of the patches to consider (rows, cols)
        patch_stride: the shift between two consecutive patches, default:patch_size
    """

    def __init__(
        self,
        root: str,
        transform=None,
        patch_size: tuple = (128, 128),
        patch_stride: tuple = None,
    ):
        self.root = root

        # alos2_url = "https://ietr-lab.univ-rennes1.fr/polsarpro-bio/san-francisco/dataset/SAN_FRANCISCO_ALOS2.zip"
        # labels_url = "https://raw.githubusercontent.com/liuxuvip/PolSF/master/SF-ALOS2/SF-ALOS2-label2d.png"

        crop_coordinates = ((2832, 736), (7888, 3520))
        self.alos_dataset = ALOSDataset(
            root, transform, crop_coordinates, patch_size, patch_stride
        )
        if isinstance(root, str):
            root = pathlib.Path(root)
        self.labels = np.array(Image.open(root.parent / "SF-ALOS2-label2d.png"))

    def __len__(self):
        # Return the number of samples in dataset
        return len(self.alos_dataset)

    def __getitem__(self, idx):
        alos_patch = self.alos_dataset[idx]
        # TODO: get the labels
        labels = None

        return alos_patch, labels


# def download_zip(dir, dataset_url):
#     # Check if download is necessary
#     if not os.path.exists(dir):
#         os.makedirs(dir)

#     zip_path = os.path.join(dir, "polsf_dataset.zip")

#     # Downloading the dataset
#     if not os.path.exists(zip_path):
#         print("Downloading dataset...")
#         response = requests.get(dataset_url, stream=True)

#         total_size = int(response.headers.get("content-length", 0))
#         block_size = 1024  # 1 Kibibyte
#         t = tqdm(total=total_size, unit="iB", unit_scale=True)

#         with open(zip_path, "wb") as file:
#             for data in response.iter_content(block_size):
#                 t.update(len(data))
#                 file.write(data)
#         t.close()

#         if total_size != 0 and t.n != total_size:
#             print("ERROR: Something went wrong")

#     # Extract the dataset
#     try:
#         with zipfile.ZipFile(zip_path, "r") as zip_ref:
#             print("Extracting dataset...")
#             file_list = zip_ref.namelist()
#             with tqdm(
#                 total=len(file_list), unit="file", desc="Extracting files"
#             ) as pbar:
#                 for file in file_list:
#                     try:
#                         zip_ref.extract(file, dir)
#                     except zipfile.BadZipFile as e:
#                         print(f"Error extracting {file}: {e}")
#                     pbar.update(1)
#     except zipfile.BadZipFile as e:
#         print(f"Error with the zip archive: {e}")
#         return

#     # Delete the zip file if no longer needed
#     # os.remove(zip_path)

#     # Load your data here (after extraction)


# def slide_and_save_crops(
#     array_3d: np.array,
#     output_folder: str,
#     crop_size: tuple = (128, 128),
#     step_size: int = 16,
# ):
#     r"""
#     Slides over a 3D numpy array, extracts $128\times128$ crops, and saves them with a progress bar.

#     Arguments:
#         array_3d: Input 3D numpy array.
#         output_folder: Folder where cropped images will be saved.
#         crop_size: Size of each crop (height, width).
#         step_size: Step size for sliding the window.
#     """
#     # Check if the directory exists
#     if os.path.exists(output_folder) and os.path.isdir(output_folder):
#         # Remove the directory and all its contents
#         shutil.rmtree(output_folder)
#         print(f"The directory {output_folder} has been removed.")

#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#         print(f"The directory {output_folder} has been created.")

#     _, height, width = array_3d.shape
#     total_crops = ((height - crop_size[0]) // step_size + 1) * (
#         (width - crop_size[1]) // step_size + 1
#     )

#     with tqdm(total=total_crops, desc="Processing Crops", unit="crop") as pbar:
#         for y in range(0, height - crop_size[0] + 1, step_size):
#             for x in range(0, width - crop_size[1] + 1, step_size):
#                 crop = torch.from_numpy(
#                     array_3d[:, y : y + crop_size[0], x : x + crop_size[1]]
#                 )
#                 torch.save(
#                     crop,
#                     os.path.join(os.path.join(output_folder, f"crop_y{y}_x{x}.pt")),
#                 )
#                 pbar.update(1)
