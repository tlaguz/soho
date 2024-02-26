import numpy as np
import torch

from masker.normalizers.normalizer import Normalizer


class ExposureNormalizer(Normalizer):
    def __init__(self):
        pass

    def normalize_observation(self, image, header):
        exptime = float(header['EXPTIME'])/10.0 * 100
        return image/exptime

    def normalize_diff(self, diff, header):
        return diff

    def normalize_label(self, label):
        return label
