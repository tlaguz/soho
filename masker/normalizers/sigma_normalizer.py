import numpy as np
import torch

from masker.normalizers.normalizer import Normalizer


class SigmaNormalizer(Normalizer):
    def __init__(self):
        pass

    def normalize_observation(self, image, header):
        #datasig = float(header['DATASIG'])
        #dataavg = float(header['DATAAVG'])

        bottom_p = np.percentile(image, 5)
        top_p = np.percentile(image, 95)
        filtered_image = image[(image > bottom_p) & (image < top_p)]

        mean = np.mean(filtered_image)
        std = np.std(filtered_image)

        return (image-mean)/std * 10

    def normalize_diff(self, diff, header):
        return diff

    def normalize_label(self, label):
        return label
