import numpy as np
import torch

from masker.normalizers.normalizer import Normalizer


class BackgroundNormalizer(Normalizer):
    def __init__(self, box_size=128):
        self.box_size = box_size

    def _get_background_level(self, image):
        h, w = image.shape[0], image.shape[1]
        box_size = self.box_size

        # Define corners' locations
        boxes = [
            image[0:box_size, 0:box_size],  # Top-left box
            image[0:box_size, w - box_size:w],  # Top-right box
            image[h - box_size:h, 0:box_size],  # Bottom-left box
            image[h - box_size:h, w - box_size:w]  # Bottom-right box
        ]

        per_box = [np.mean(box) for box in boxes]

        return np.median(per_box)

    def normalize_observation(self, image, header):
        bg_level = self._get_background_level(image)
        if bg_level == 0:
            return image
        return image/bg_level

    def normalize_diff(self, diff, header):
        bg_level = self._get_background_level(diff)
        if bg_level == 0:
            return diff
        return diff - bg_level

    def normalize_label(self, label):
        return label
