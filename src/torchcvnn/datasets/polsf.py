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

# External imports
import numpy as np
import requests
import spectral.io.envi as envi
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset


class PolSFAlos2Dataset(Dataset):
    def __init__(
        self,
        root=None,
        transform=None,
        save=False,
        coordinates=None,
        download=False,
        url="https://ietr-lab.univ-rennes1.fr/polsarpro-bio/san-francisco/dataset/SAN_FRANCISCO_ALOS2.zip",
    ):
        """
        Args:
            coordinates: the labelised crop of the image is given by the following coordinates(2832, 736, 7888, 3520)
        """
        self.root = root
        if not self.root:
            self.root = os.path.join(os.getcwd(), "datasets/SAN_FRANCISCO_ALOS2")

        self.transform = transform
        self.img_dir = os.path.join(self.root, "imgs")
        if download:
            download_zip(dataset_url=url)
        if save:
            polar = self.load(coordinates)
            slide_and_save_crops(array_3d=polar, output_folder=self.img_dir)

        self.files_names = [
            f
            for f in os.listdir(self.img_dir)
            if os.path.isfile(os.path.join(self.img_dir, f))
        ]
        self.files_names.sort()

    def __len__(self):
        # Return the number of samples in dataset
        return len(self.files_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.files_names[idx])

        image = torch.load(img_name)

        if self.transform:
            image = self.transform(image)

        return image

    def load(self, coordinates):
        s_11_meta = envi.open(self.root + "/s11.bin.hdr", self.root + "/s11.bin")
        s_12_meta = envi.open(self.root + "/s12.bin.hdr", self.root + "/s12.bin")
        s_21_meta = envi.open(self.root + "/s21.bin.hdr", self.root + "/s21.bin")
        s_22_meta = envi.open(self.root + "/s22.bin.hdr", self.root + "/s22.bin")

        s_11 = s_11_meta.read_band(0)
        s_12 = s_12_meta.read_band(0)
        s_21 = s_21_meta.read_band(0)
        s_22 = s_22_meta.read_band(0)

        if coordinates is not None:
            assert isinstance(coordinates, tuple) and len(coordinates) == 4
            x1, y1, x2, y2 = coordinates
            s_11 = s_11[x1:x2, y1:y2]
            s_12 = s_12[x1:x2, y1:y2]
            s_21 = s_21[x1:x2, y1:y2]
            s_22 = s_22[x1:x2, y1:y2]

        assert np.all(s_21 == s_12)

        stack = (1 / np.sqrt(2)) * np.stack(
            (s_11 - s_22, 2 * s_12, s_11 + s_22), axis=-1
        ).astype(np.complex64).transpose(2, 0, 1)
        return stack


def download_zip(dir, dataset_url):
    # Check if download is necessary
    if not os.path.exists(dir):
        os.makedirs(dir)

    zip_path = os.path.join(dir, "polsf_dataset.zip")

    # Downloading the dataset
    if not os.path.exists(zip_path):
        print("Downloading dataset...")
        response = requests.get(dataset_url, stream=True)

        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        t = tqdm(total=total_size, unit="iB", unit_scale=True)

        with open(zip_path, "wb") as file:
            for data in response.iter_content(block_size):
                t.update(len(data))
                file.write(data)
        t.close()

        if total_size != 0 and t.n != total_size:
            print("ERROR: Something went wrong")

    # Extract the dataset
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            print("Extracting dataset...")
            file_list = zip_ref.namelist()
            with tqdm(
                total=len(file_list), unit="file", desc="Extracting files"
            ) as pbar:
                for file in file_list:
                    try:
                        zip_ref.extract(file, dir)
                    except zipfile.BadZipFile as e:
                        print(f"Error extracting {file}: {e}")
                    pbar.update(1)
    except zipfile.BadZipFile as e:
        print(f"Error with the zip archive: {e}")
        return

    # Delete the zip file if no longer needed
    # os.remove(zip_path)

    # Load your data here (after extraction)


def slide_and_save_crops(array_3d, output_folder, crop_size=(128, 128), step_size=16):
    """
    Slides over a 3D numpy array, extracts 128x128 crops, and saves them with a progress bar.

    :param array_3d: Input 3D numpy array.
    :param output_folder: Folder where cropped images will be saved.
    :param crop_size: Size of each crop (height, width).
    :param step_size: Step size for sliding the window.
    """
    # Check if the directory exists
    if os.path.exists(output_folder) and os.path.isdir(output_folder):
        # Remove the directory and all its contents
        shutil.rmtree(output_folder)
        print(f"The directory {output_folder} has been removed.")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"The directory {output_folder} has been created.")

    _, height, width = array_3d.shape
    total_crops = ((height - crop_size[0]) // step_size + 1) * (
        (width - crop_size[1]) // step_size + 1
    )

    with tqdm(total=total_crops, desc="Processing Crops", unit="crop") as pbar:
        for y in range(0, height - crop_size[0] + 1, step_size):
            for x in range(0, width - crop_size[1] + 1, step_size):
                crop = torch.from_numpy(
                    array_3d[:, y : y + crop_size[0], x : x + crop_size[1]]
                )
                torch.save(
                    crop,
                    os.path.join(os.path.join(output_folder, f"crop_y{y}_x{x}.pt")),
                )
                pbar.update(1)
