import numpy as np

from masker.labelers.labeler import Labeler
from masker.utils import rotate_by_fits_header, create_gaussian, parameters_to_string


class GaussLabeler(Labeler):
    def __init__(self, height=1024, width=1024, sigma=15.0):
        super().__init__(height, width)
        self.sigma = sigma

    def get_cache_name(self):
        params = [self.height, self.width, self.sigma]
        params_txt = parameters_to_string(params)
        return f"gauss{params_txt}"

    def label(self, rows, cols, fits_header):
        label = np.zeros((self.height, self.width))

        for i in range(len(rows)):
            coordinate_y = rows[i]
            coordinate_x = cols[i]
            coordinate_x, coordinate_y = rotate_by_fits_header(coordinate_x, coordinate_y, fits_header)
            label += create_gaussian(self.height, self.width, coordinate_y, coordinate_x, self.sigma)

        if np.max(label) != 0:
            label /= np.max(label)

        return label
