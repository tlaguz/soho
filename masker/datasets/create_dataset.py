import torch
import torch.utils.data

from masker.datasets.augmented_dataset import AugmentedDataset
from masker.datasets.cached_dataset import CachedDataset
from masker.datasets.dataset import FitsDataset
from masker.datasets.video_dataset import FitsVideoDataset
from masker.utils import get_paths


def create_dataset(dbconn, dtype="fp32", augment=True):
    paths = get_paths()

    ds = FitsDataset(dbconn, running_diff=True, mask_disk=False, dtype=dtype)
    ds = torch.utils.data.Subset(ds, list(range(100000, 100000+5000)))
    ds = CachedDataset(ds, paths.train_cache)
    if augment:
        ds = AugmentedDataset(ds)
    return ds

def create_validation_dataset(dbconn, dtype="fp32", augment=True):
    paths = get_paths()
    ds = FitsDataset(dbconn, running_diff=True, mask_disk=True, dtype=dtype)
    ds = torch.utils.data.Subset(ds, list(range(110000, 110000+250)))
    ds = CachedDataset(ds, paths.valid_cache)
    if augment:
        ds = AugmentedDataset(ds)
    return ds
