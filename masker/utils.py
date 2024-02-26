from dataclasses import dataclass

import numpy as np
from skimage import exposure

@dataclass
class Paths:
    db_file: str
    yhts_directory: str
    fits_directory: str
    save_dir: str
    train_cache: str
    valid_cache: str
    fits_cache: str
    repository_cache: str


def get_paths():
    return Paths(
        db_file = "/home/tlaguz/db.sqlite3",
        yhts_directory = "/mnt/mlpool/yhts/",
        fits_directory = "/mnt/mlpool/soho_seiji/",
        save_dir = '/mnt/mlpool/soho_seiji/checkpoints/',
        train_cache = '/mnt/trunk/tlaguz/soho/train_cache/',
        valid_cache = '/mnt/trunk/tlaguz/soho/valid_cache/',
        fits_cache = '/mnt/trunk/tlaguz/soho/fits_cache/',
        repository_cache = '/mnt/trunk/tlaguz/soho/repository_cache/'
    )

def normalize_by_statistics(image, bot = 2, top = 99):
    # Convert to float
    image = image
    botp, topp = np.percentile(image, (bot, top))

    if bot == 0:
        botp = np.min(image)

    if top == 100:
        topp = np.max(image)

    image = exposure.rescale_intensity(image, in_range=(botp, topp))
    return image


def get_corner_pixels(image, box_size=32):
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

    return corner_pixels

def normalize_by_corner_statistics(image, box_size=32, bot = 2, top = 99):
    # Convert to float
    image = image
    h, w = image.shape[0], image.shape[1]

    corner_pixels = get_corner_pixels(image, box_size)

    # Compute percentiles on the gathered pixels
    botp = np.percentile(corner_pixels, bot)
    topp = np.percentile(image, top)

    if top == 100:
        topp = np.max(image)

    # Rescale intensities based on computed percentiles
    image = exposure.rescale_intensity(image, in_range=(botp, topp))

    corner_pixels = get_corner_pixels(image, box_size)

    # Background around 0
    avg = np.median(corner_pixels)
    image -= avg             # shift the mean to 0
    min, max = image.min(), image.max()
    if max - min != 0:
        image /= max - min   # scale values by the range

    #image = 2*image - 1

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


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib
#matplotlib.use("TkAgg")

def plot_animate(height=1024, width=1024, y=512, x=512, sigma_r=10, sigma_phi=10):
    # Create a new figure
    fig, ax = plt.subplots()

    # Initialize a list to store the artists for each frame
    imgs = []

    frames = 100
    for i in range(frames):
        # Compute the gaussian for the updated x
        gaussian = warped_polar_gauss(height, width, y, x+i*(512/frames), sigma_r, sigma_phi)

        # Create the image artist with the colorbar and append to the list
        im = ax.imshow(gaussian, origin='lower', animated=True)

        # Add the artists for this frame to imgs
        imgs.append([im])

    # Create the animation
    ani = animation.ArtistAnimation(fig, imgs, interval=50, blit=True)

    # Show the plot
    plt.show()

def circle_diff(v1, v2):
    # This function will handle the difference between two angles properly
    d = v1 - v2
    return np.arctan2(np.sin(d), np.cos(d))

def gauss_arbitrary_variable(grid, v, sigma, circular=False):
    diff = grid - v
    if circular:
        diff = circle_diff(grid, v)

    return np.exp(-(diff ** 2) / (2 * sigma ** 2))

def warped_polar_gauss(height, width, y, x, sigma_r, sigma_phi):
    x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))

    r_grid = np.sqrt((x_grid - width/2) ** 2 + (y_grid - height/2) ** 2)

    # Using atan2 to get the correct quadrant
    phi_grid = np.arctan2(y_grid - height/2, x_grid - width/2) % (2 * np.pi)

    # We take the difference of the vectors for radius and angles
    # In this case, r and phi are considered as coordinates of center of image.
    r = np.sqrt((height / 2 - y) ** 2 + (width / 2 - x) ** 2)
    phi = np.arctan2(y - height / 2, x - width / 2) % (2 * np.pi)

    gaussian_r = gauss_arbitrary_variable(r_grid, r, sigma_r)

    gaussian_phi = gauss_arbitrary_variable(phi_grid, phi, sigma_phi, circular=True)

    g = gaussian_r * gaussian_phi
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
