from masker.normalizers.normalizer import Normalizer


class ComboNormalizer(Normalizer):
    def __init__(self, normalizers : [Normalizer]):
        self.normalizers = normalizers

    def normalize_observation(self, image, header):
        for normalizer in self.normalizers:
            image = normalizer.normalize_observation(image, header)
        return image

    def normalize_diff(self, diff, header):
        for normalizer in self.normalizers:
            diff = normalizer.normalize_diff(diff, header)
        return diff

    def normalize_label(self, label):
        for normalizer in self.normalizers:
            label = normalizer.normalize_label(label)
        return label
