import collections
from dataclasses import dataclass

import numpy as np
from astropy.io import fits
from torch.utils.data import Dataset

from masker.fits_loader import FitsLoader
from masker.repositories.training_points_repository import TrainingPointsRepository
from masker.trainer_config import create_labeler, create_normalizer
from masker.utils import create_masking_circle, parameters_to_string


@dataclass
class FitsDatasetDto:
    image : np.ndarray
    image_header : fits.header.Header
    image_prev : np.ndarray
    image_prev_header : fits.header.Header
    label : np.ndarray
    txt : str

class FitsDataset(Dataset):
    def __init__(self, dbconn, running_diff=False, mask_disk=False, detector='C2', dtype="fp32"):
        if dtype == "bf16":
            self.dtype = np.float16
        elif dtype == "fp16":
            self.dtype = np.float16
        else:
            self.dtype = np.float32
        self.detector = detector
        self.mask_disk = mask_disk
        self.fits_loader = FitsLoader()
        self.dbconn = dbconn
        self.training_points_repository = TrainingPointsRepository(dbconn)
        self.labeler = create_labeler()

    def get_cache_name(self):
        params = [self.mask_disk, self.detector]
        params_txt = parameters_to_string(params)
        return f"rdiff_{params_txt}_{self.labeler.get_cache_name()}"

    def __len__(self):
        return self.training_points_repository.get_training_data_len(self.detector)

    def __getitem__(self, idx):
        if isinstance(idx, collections.abc.Iterable):
            raise ValueError("Indexing with an iterable is not supported")

        record = self.training_points_repository.get_training_data_by_id(idx, self.detector)
        if record is None:
            pass

        txt = record["filename"] + "  -  " + record["filename_prev"] + "  -  " + record["filename_yht"]

        hdul = self.fits_loader.get_cached_fits(record["filename"])
        if hdul is None:
            print("Error loading fits file " + record["filename"] + " for id " + str(idx - 1))
            image = np.zeros((1024, 1024), dtype=self.dtype)
        else:
            image = hdul[0].data.astype(np.float32)

        hdul_prev = self.fits_loader.get_cached_fits(record["filename_prev"])
        if hdul_prev is None:
            print("Error loading fits file " + record["filename_prev"] + " for id " + str(idx - 1))
            image_prev = np.zeros((1024, 1024), dtype=self.dtype)
        else:
            image_prev = hdul_prev[0].data.astype(np.float32)

        if self.mask_disk:
            radius = 180
            if record["detector"] == 'C3':
                radius = 2

            mask = create_masking_circle(1024, 1024, radius, hdul[0].header)
            image[mask] = 0

        rows = list(map(int, record["aggregated_row"].split(",")))
        cols = list(map(int, record["aggregated_col"].split(",")))

        label = self.labeler.label(rows, cols, hdul[0].header)
        if np.max(label) == 0:
            print("Label is broken for id " + str(idx - 1))

        return FitsDatasetDto(image, hdul[0].header, image_prev, hdul_prev[0].header, label, txt)
