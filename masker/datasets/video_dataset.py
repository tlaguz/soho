import collections

import numpy as np
from astropy.io import fits
from torch.utils.data import Dataset

from masker.fits_loader import FitsLoader
from masker.labelers.create_labeler import create_labeler
from masker.normalizers.create_normalizer import create_normalizer
from masker.repositories.fits_repository import FitsRepository, FitsDto
from masker.repositories.training_points_repository import TrainingPointsRepository
from masker.utils import create_masking_circle, rotate_image_by_fits_header


class FitsVideoDataset(Dataset):
    def __init__(self, dbconn, running_diff=False, mask_disk=False, detector='C2', dtype="fp32", frames=6):
        if frames < 1:
            raise ValueError("frames must be at least 1")

        self.frames = frames
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
        self.fits_repository = FitsRepository(dbconn)
        self.labeler = create_labeler()
        self.normalizer = create_normalizer()

    def get_cache_name(self):
        return f"running_diff_video_{self.frames}"

    def __len__(self):
        return self.training_points_repository.get_training_data_len(self.detector)

    def prepare_frame(self, idx, fits : FitsDto, prev_fits : FitsDto):
        hdul = self.fits_loader.get_cached_fits(fits.filename)
        if hdul is None:
            print("Error loading fits file " + fits.filename + " for id " + str(idx - 1))
            return np.zeros((1024, 1024), dtype=self.dtype)
        else:
            image = hdul[0].data.astype(np.float32)
            image = self.normalizer.normalize_observation(image, hdul[0].header)
            image = rotate_image_by_fits_header(image, hdul[0].header)

        if self.running_diff:
            hdul_prev = self.fits_loader.get_cached_fits(prev_fits.filename)
            if hdul_prev is None:
                print("Error loading fits file " + prev_fits.filename + " for id " + str(idx - 1))
                return np.zeros((1024, 1024), dtype=self.dtype)

            image_prev = hdul_prev[0].data.astype(np.float32)
            image_prev = self.normalizer.normalize_observation(image_prev, hdul_prev[0].header)
            image_prev = rotate_image_by_fits_header(image_prev, hdul_prev[0].header)
            image = image - image_prev
            image = self.normalizer.normalize_diff(image, hdul[0].header)

        if self.mask_disk:
            radius = 180
            if fits.detector == 'C3':
                radius = 2

            mask = create_masking_circle(1024, 1024, radius, hdul[0].header)
            image[mask] = 0

        return image
    def __getitem__(self, idx):
        if isinstance(idx, collections.abc.Iterable):
            raise ValueError("Indexing with an iterable is not supported")

        record = self.training_points_repository.get_training_data_by_id(idx, self.detector)
        if record is None:
            pass

        txt = record["filename_yht"]
        current_filename = record["filename"]
        image = []

        for i in range(0, self.frames):
            current_fits = self.fits_repository.get_fits(current_filename)
            previous_fits = self.fits_repository.get_previous_fits(current_filename)
            img = self.prepare_frame(idx, current_fits, previous_fits)
            image.append(img)
            current_filename = previous_fits.filename

        hdul = self.fits_loader.get_cached_fits(record["filename"])

        rows = list(map(int, record["aggregated_row"].split(",")))
        cols = list(map(int, record["aggregated_col"].split(",")))

        label = self.labeler.label(rows, cols, hdul[0].header)
        label = self.normalizer.normalize_label(label)
        if np.max(label) == 0:
            print("Label is broken for id " + str(idx - 1))

        return image, label, txt
