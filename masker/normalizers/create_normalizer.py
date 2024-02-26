from masker.normalizers.background_normalizer import BackgroundNormalizer
from masker.normalizers.exposure_normalizer import ExposureNormalizer


def create_normalizer():
    #return BackgroundNormalizer()
    return ExposureNormalizer()