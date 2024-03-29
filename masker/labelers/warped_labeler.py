import numpy as np

from masker.labelers.labeler import Labeler
from masker.utils import rotate_by_fits_header, warped_polar_gauss, parameters_to_string


class WarpedLabeler(Labeler):
    def __init__(self, height=1024, width=1024, sigma_r=5, sigma_phi=0.05):
        super().__init__(height, width)
        self.sigma_r = sigma_r
        self.sigma_phi = sigma_phi

    def get_cache_name(self):
        params = [self.height, self.width, self.sigma_r, self.sigma_phi]
        params_txt = parameters_to_string(params)
        return f"gauss{params_txt}"

    def label(self, rows, cols, fits_header):
        label = np.zeros((self.height, self.width))

        for i in range(len(rows)):
            coordinate_y = rows[i]
            coordinate_x = cols[i]
            coordinate_x, coordinate_y = rotate_by_fits_header(coordinate_x, coordinate_y, fits_header)
            label += warped_polar_gauss(self.height, self.width, coordinate_y, coordinate_x, self.sigma_r, self.sigma_phi)

        if np.max(label) != 0:
            label /= np.max(label)

        return label
