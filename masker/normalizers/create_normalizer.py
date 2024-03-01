from masker.normalizers.background_normalizer import BackgroundNormalizer
from masker.normalizers.combo_normalizer import ComboNormalizer
from masker.normalizers.exposure_normalizer import ExposureNormalizer
from masker.normalizers.normalizer import Normalizer
from masker.normalizers.sigma_normalizer import SigmaNormalizer


def create_normalizer():
    return SigmaNormalizer()

    return BackgroundNormalizer()

    return Normalizer()

    return ComboNormalizer([SigmaNormalizer(), BackgroundNormalizer()])

    #return ExposureNormalizer()
