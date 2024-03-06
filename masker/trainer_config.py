import torch
import torch.utils.data

def create_dataset(dbconn, dtype="fp32", augment=True):
    from masker.datasets.fits_dataset import FitsDataset
    from masker.datasets.video_dataset import FitsVideoDataset
    from masker.datasets.augmented_dataset import AugmentedDataset
    from masker.datasets.cached_dataset import CachedDataset
    from masker.utils import get_paths

    paths = get_paths()

    ds = FitsDataset(dbconn, running_diff=True, mask_disk=False, dtype=dtype)
    ds = torch.utils.data.Subset(ds, list(range(100000, 100000+5000)))
    ds = CachedDataset(ds, paths.train_cache)
    if augment:
        ds = AugmentedDataset(ds)
    return ds

def create_validation_dataset(dbconn, dtype="fp32", augment=True):
    from masker.datasets.fits_dataset import FitsDataset
    from masker.datasets.video_dataset import FitsVideoDataset
    from masker.datasets.augmented_dataset import AugmentedDataset
    from masker.datasets.cached_dataset import CachedDataset
    from masker.utils import get_paths

    paths = get_paths()
    ds = FitsDataset(dbconn, running_diff=True, mask_disk=False, dtype=dtype)
    ds = torch.utils.data.Subset(ds, list(range(110000, 110000+250)))
    ds = CachedDataset(ds, paths.valid_cache)
    if augment:
        ds = AugmentedDataset(ds)
    return ds

def create_labeler():
    from masker.labelers.warped_labeler import WarpedLabeler

    return WarpedLabeler(height = 1024, width = 1024, sigma_r = 8, sigma_phi = 0.06)

def create_loss(device = None):
    from masker.losses.combined_loss import combined_loss

    #loss = F.binary_cross_entropy_with_logits
    loss = lambda input, target: combined_loss(input, target, 0.5)
    return loss

def create_model():
    from masker.models.unet import UNetWrapper
    from masker.models.segformer import SegFormerWrapper

    wrapper = SegFormerWrapper(channels=1)
    #wrapper = UNetWrapper()

    return wrapper


def create_normalizer():
    from masker.normalizers.sigma_normalizer import SigmaNormalizer

    return SigmaNormalizer()

    return BackgroundNormalizer()

    return Normalizer()

    return ComboNormalizer([SigmaNormalizer(), BackgroundNormalizer()])

    #return ExposureNormalizer()
