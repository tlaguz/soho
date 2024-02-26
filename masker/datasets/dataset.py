import collections

import numpy as np
from astropy.io import fits
from torch.utils.data import Dataset

from masker.fits_loader import FitsLoader
from masker.labelers.create_labeler import create_labeler
from masker.normalizers.create_normalizer import create_normalizer
from masker.repositories.training_points_repository import TrainingPointsRepository
from masker.utils import create_masking_circle


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
        self.running_diff = running_diff
        self.fits_loader = FitsLoader()
        self.dbconn = dbconn
        self.training_points_repository = TrainingPointsRepository(dbconn)
        self.labeler = create_labeler()
        self.normalizer = create_normalizer()

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
            return np.zeros((1024, 1024), dtype=self.dtype), np.zeros((1024, 1024), dtype=self.dtype), txt
        else:
            image = hdul[0].data.astype(np.float32)
            image = self.normalizer.normalize_observation(image, hdul[0].header)

        if self.running_diff:
            hdul_prev = self.fits_loader.get_cached_fits(record["filename_prev"])
            if hdul_prev is None:
                print("Error loading fits file " + record["filename_prev"] + " for id " + str(idx - 1))
                return np.zeros((1024, 1024), dtype=self.dtype), np.zeros((1024, 1024), dtype=self.dtype), txt

            image_prev = hdul_prev[0].data.astype(np.float32)
            image_prev = self.normalizer.normalize_observation(image_prev, hdul_prev[0].header)
            image = image - image_prev
            image = self.normalizer.normalize_diff(image, hdul[0].header)

        if self.mask_disk:
            radius = 180
            if record["detector"] == 'C3':
                radius = 2

            mask = create_masking_circle(1024, 1024, radius, hdul[0].header)
            image[mask] = 0

        rows = list(map(int, record["aggregated_row"].split(",")))
        cols = list(map(int, record["aggregated_col"].split(",")))

        label = self.labeler.label(rows, cols, hdul[0].header)
        label = self.normalizer.normalize_label(label)
        if np.max(label) == 0:
            print("Label is broken for id " + str(idx - 1))

        return image, label, txt
