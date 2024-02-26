import numpy as np

from masker.utils import rotate_by_fits_header, create_gaussian


class Labeler:
    def __init__(self, height = 1024, width = 1024, sigma = 15.0):
        self.sigma = sigma
        self.height = height
        self.width = width

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
