import collections
import os
import pickle
from torch.utils.data import Dataset
import torch.utils.data

class CachedDataset(Dataset):
    def __init__(self,
                 dataset: Dataset,
                 cache_directory: str,
                 ):
        self.dataset = dataset
        self.cache_directory = cache_directory + "/" + self.get_cache_name()

        os.makedirs(self.cache_directory, exist_ok=True)

    def get_cache_name(self):
        if isinstance(self.dataset, torch.utils.data.Subset):
            cn = self.dataset.dataset.get_cache_name()
        else:
            cn = self.dataset.get_cache_name()

        return "cached_" + cn

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(idx, collections.abc.Iterable):
            raise ValueError("Indexing with an iterable is not supported")

        cache_file = os.path.join(self.cache_directory, f"{idx}.pkl")

        # If cached, load from cache
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        else:
            # If not in cache, get the data and cache it
            data = self.dataset[idx]
            with open(cache_file, "wb") as f:
                pickle.dump(data, f)
            return data