import numpy as np

from masker.utils import parameters_to_string


class Labeler:
    def __init__(self, height=1024, width=1024):
        self.height = height
        self.width = width

    def get_cache_name(self):
        params = [self.height, self.width]
        params_txt = parameters_to_string(params)
        return f"labeler{params_txt}"

    def label(self, rows, cols, fits_header):
        return np.zeros((self.height, self.width))
