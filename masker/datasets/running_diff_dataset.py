import numpy as np
import torch
from torch.utils.data import Dataset

from masker.trainer_config import create_normalizer


class RunningDiffDataset(Dataset):
    def __init__(self,
                 dataset: torch.utils.data.Dataset,
                 randomize_exposure: bool = False):
        self.randomize_exposure = randomize_exposure
        self.dataset = dataset
        self.normalizer = create_normalizer()

    def _get_exposure_factor(self):
        if not self.randomize_exposure:
            return 1

        mean = 1
        std_dev = 0.007

        return np.random.normal(mean, std_dev)

    def __getitem__(self, idx: int):
        dto = self.dataset[idx] #dto is FitsDatasetDto
        img = self.normalizer.normalize_observation(dto.image, dto.image_header)
        img_prev = self.normalizer.normalize_observation(dto.image_prev, dto.image_prev_header)
        label = self.normalizer.normalize_label(dto.label)
        txt = dto.txt

        img_prev = img_prev * self._get_exposure_factor()

        diff = img - img_prev
        diff = self.normalizer.normalize_diff(diff, dto.image_header)

        return diff, label, txt

    def __len__(self):
        return len(self.dataset)
