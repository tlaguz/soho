import numpy as np
import torch
from torch.utils.data import Dataset

class AugmentedDataset(Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset):
        self.dataset = dataset

    def __getitem__(self, idx: int):
        base_idx = idx // 8
        transformed_idx = idx % 8

        img, label, txt = self.dataset[base_idx]

        if transformed_idx < 4:
            # Rotate the image
            img = np.rot90(img, k=transformed_idx, axes=(1, 0)).copy()
            label = np.rot90(label, k=transformed_idx, axes=(1, 0)).copy()
        else:
            # Flip the image horizontally and then rotate
            img = np.flipud(img)
            label = np.flipud(label)
            img = np.rot90(img, k=(transformed_idx-4), axes=(1, 0)).copy()
            label = np.rot90(label, k=(transformed_idx-4), axes=(1, 0)).copy()

        return img, label, txt

    def __len__(self):
        return len(self.dataset) * 8
