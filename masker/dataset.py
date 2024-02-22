import numpy as np
import torch
from astropy.io import fits
from torch.utils.data import Dataset

from masker.labeler import Labeler
from masker.repositories.training_points_repository import TrainingPointsRepository
from masker.utils import normalize_by_statistics, rotate_by_fits_header, create_masking_circle, create_gaussian, \
    normalize_by_corner_statistics


class FitsDataset(Dataset):
    def __init__(self, dbconn, fits_directory, transform=None, target_transform=None, running_diff=False, mask_disk=False, detector='C2', dtype="fp32"):
        if dtype == "bf16":
            self.dtype = np.float16
        elif dtype == "fp16":
            self.dtype = np.float16
        else:
            self.dtype = np.float32
        self.detector = detector
        self.mask_disk = mask_disk
        self.running_diff = running_diff
        self.fits_directory = fits_directory
        self.dbconn = dbconn
        self.training_points_repository = TrainingPointsRepository(dbconn)
        self.transform = transform
        self.target_transform = target_transform
        self.labeler = Labeler()

    def __len__(self):
        return 5000
        return self.training_points_repository.get_training_data_len(self.detector)

    def __getitem__(self, idx):
        idx = idx + 1 + 100000

        record = self.training_points_repository.get_training_data_by_id(idx, self.detector)
        if record is None:
            pass

        hdul = fits.open(self.fits_directory + record["filename"])
        image = hdul[0].data.astype(np.float32)
        image = normalize_by_corner_statistics(image)

        if self.running_diff:
            hdul_prev = fits.open(self.fits_directory + record["filename_prev"])
            image_prev = hdul_prev[0].data.astype(np.float32)
            image_prev = normalize_by_corner_statistics(image_prev)
            image = image - image_prev
            image = normalize_by_corner_statistics(image)

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

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, record["filename"] + "  -  " + record["filename_prev"] + "  -  " + record["filename_yht"]
