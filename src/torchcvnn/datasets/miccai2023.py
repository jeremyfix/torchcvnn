# coding: utf-8

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
from enum import Enum
import pathlib
import logging

# External imports
from torch.utils.data import Dataset
import scipy.io


class CINEView(Enum):
    SAX = 1
    LAX = 2


class AccFactor(Enum):
    ACC4 = 4
    ACC8 = 8
    ACC10 = 10


class MICCAI2023(Dataset):
    """
    Loads the MICCAI2023 challenge data for the reconstruction task Task 1

    The data are described on https://cmrxrecon.github.io/Task1-Cine-reconstruction.html

    You need to download the data before hand in order to use this class.

    The structure of the dataset is as follows:

        rootdir/ChallengeData/MultiCoil/cine/TrainingSet/P{id}/
                                    - cine_sax.mat
                                    - cin_lax.mat
        rootdir/ChallengeData/MultiCoil/cine/TrainingSet/AccFactor04/P{id}/
                                    - cine_sax.mat
                                    - cine_sax_mask.mat
                                    - cin_lax.mat
                                    - cine_lax_mask.mat
        rootdir/ChallengeData/MultiCoil/cine/TrainingSet/AccFactor08/P{id}/
                                    - cine_sax.mat
                                    - cine_sax_mask.mat
                                    - cin_lax.mat
                                    - cine_lax_mask.mat
        rootdir/ChallengeData/MultiCoil/cine/TrainingSet/AccFactor10/P{id}/
                                    - cine_sax.mat
                                    - cine_sax_mask.mat
                                    - cin_lax.mat
                                    - cine_lax_mask.mat

    There are also the Single-Coil data which is not yet considered by this implementation

    This is a recontruction dataset. The goal is to reconstruct the fully sampled k-space
    from the subsampled k-space. The acceleratation factor specifies the subsampling rate.
    """

    def __init__(
        self,
        rootdir: str,
        view: CINEView = CINEView.SAX,
        transform=None,
        acc_factor: AccFactor = AccFactor.ACC4,
    ):
        self.rootdir = pathlib.Path(rootdir)
        self.transform = transform

        if view == CINEView.SAX:
            self.input_filename = "cine_sax.mat"
            self.mask_filename = "cine_sax_mask.mat"
        elif view == CINEView.LAX:
            self.input_filename = "cine_lax.mat"
            self.mask_filename = "cine_lax_mask.mat"

        # List all the available data
        self.fullsampled_rootdir = self.rootdir / "MultiCoil" / "cine" / "TrainingSet"
        self.subsampled_rootdir = (
            self.rootdir
            / "MultiCoil"
            / "cine"
            / "TrainingSet"
            / f"AccFactor{acc_factor.value:02d}"
        )
        logging.info(f"Loading data from {self.subsampled_rootdir}")

        # We list all the patients in the subsampled data directory
        # and check we have the data, mask and full sampled data
        self.patients = []
        for patient in self.subsampled_rootdir.iterdir():
            if not patient.is_dir():
                continue

            if not (patient / self.input_filename).exists():
                logging.warning(f"Missing {self.input_filename} for patient {patient}")
                continue

            if not (patient / self.mask_filename).exists():
                logging.warning(f"Missing {self.mask_filename} for patient {patient}")
                continue

            fullsampled_patient = self.fullsampled_rootdir / patient.name
            if not (fullsampled_patient / self.input_filename).exists():
                logging.warning(
                    f"Missing {self.input_filename} for patient {fullsampled_patient}"
                )
                continue

            self.patients.append(patient)

        logging.debug(
            f"I found {len(self.patients)} patient(s) : {[p.name for p in self.patients]}"
        )

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        pass
