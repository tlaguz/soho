import numpy as np
from skimage import exposure


def normalize_by_statistics(image, bot = 2, top = 99):
    # Convert to float
    image = image
    botp, topp = np.percentile(image, (bot, top))
    image = exposure.rescale_intensity(image, in_range=(botp, topp))
    return image


def normalize_by_corner_statistics(image, box_size=32, bot = 2, top = 99):
    # Convert to float
    image = image
    h, w = image.shape[0], image.shape[1]

    # Define corners' locations
    boxes = [
        image[0:box_size, 0:box_size],  # Top-left box
        image[0:box_size, w-box_size:w],  # Top-right box
        image[h-box_size:h, 0:box_size],  # Bottom-left box
        image[h-box_size:h, w - box_size:w]  # Bottom-right box
    ]

    # Gather pixels in the corners
    corner_pixels = np.concatenate([box.ravel() for box in boxes])

    # Compute percentiles on the gathered pixels
    botp = np.percentile(corner_pixels, bot)
    topp = np.percentile(image, top)

    # Rescale intensities based on computed percentiles
    image = exposure.rescale_intensity(image, in_range=(botp, topp))
    return image


# https://www.mssl.ucl.ac.uk/grid/iau/extra/solarsoft/ssw_standards.html
def rotate_by_fits_header(i, j, header, side = 1):
    center_of_rotation_x = header["CRPIX1"]
    center_of_rotation_y = header["CRPIX2"]

    #center_of_rotation_x = 512
    #center_of_rotation_y = 512

    i = i*2
    j = j*2

    a_deg = header['CROTA1']
    a = side*np.deg2rad(a_deg)
    # rotate clockwise by a
    x = i - center_of_rotation_x
    y = j - center_of_rotation_y
    x = x * np.cos(a) + y * np.sin(a)
    y = -x * np.sin(a) + y * np.cos(a)
    x = x + center_of_rotation_x
    y = y + center_of_rotation_y
    return x, y

    # Function to generate a gaussian at a specific point
def create_gaussian(height, width, y, x, sigma):
    """Generates a Gaussian centered at (x, y) with standard deviation sigma"""
    x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))
    g = np.exp(-((x_grid - x) ** 2 + (y_grid - y) ** 2) / (2 * sigma ** 2))
    return g

def create_masking_circle(height, width, radius, header = None):
    center_x, center_y = height / 2, width / 2
    if header is not None:
        center_x = header["CRPIX1"]
        center_y = header["CRPIX2"]

    # Create index arrays for x and y
    y, x = np.ogrid[0:1024, 0:1024]

    # Create a mask for pixels within the circle
    mask = ((y - center_y) ** 2 + (x - center_x) ** 2 <= radius ** 2)

    return mask
