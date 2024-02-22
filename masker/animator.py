from datetime import datetime

import numpy as np

import PyQt5

import matplotlib
import torch

from masker.labeler import Labeler
from masker.model_inference import ModelInference
from masker.models.model_factory import create_model
from masker.repositories.fits_repository import FitsRepository
from masker.utils import normalize_by_statistics, rotate_by_fits_header, normalize_by_corner_statistics

matplotlib.use("TkAgg")
#matplotlib.use('Qt5Agg')

# Set up matplotlib
import matplotlib.pyplot as plt

from astropy.io import fits
from matplotlib import animation


class Animator:
    def __init__(self, dbconn, fits_directory):
        self.dbconn = dbconn
        self.fits_directory = fits_directory
        self.fits_repository = FitsRepository(dbconn)
        self.labeler = Labeler()

    def animate(self, yhts, model_path):
        colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white']
        c = 0
        # a list of tuples: (filename, points[], date)
        # where points is a list of tuples: (row, col, color)
        fits_points = []

        for yht in yhts:
            color = colors[c]
            c += 1
            fits_points_db = self.fits_repository.get_fits_points_by_yht_filename(yht)
            for fpd in fits_points_db:
                if fpd.detector != 'C2':
                    continue

                # add the filename to the list of filenames if it is not already there
                # if it exists already, add the point to the list of points
                found = False
                for fp in fits_points:
                    if fp[0] == fpd.filename:
                        fp[1].append((fpd.row, fpd.col, color))
                        found = True
                        break

                if not found:
                    fits_points.append((fpd.filename, [(fpd.row, fpd.col, color)], fpd.date))

        # sort the list by date
        fits_points.sort(key=lambda x: x[2])

        # instead of making a new plot for each image, make an animation with all the images
        # and the points on them
        # each frame will be a new image with the points on it
        # watch out no to add points as new frames after the image has been shown

        model_inference = None
        if model_path is not None:
            if isinstance(model_path, list):
                model_path = model_path[0]

            model_inference = ModelInference(model_path)

        fig, axs = plt.subplots(1, 4)
        axs[0].set_title('Raw with manual label')
        axs[0].norm = matplotlib.colors.LogNorm()
        axs[1].set_title('Running diff with manual label')
        #axs[1].norm = matplotlib.colors.LogNorm()
        axs[2].set_title('ML prediction on running diff')
        axs[2].norm = matplotlib.colors.LogNorm()
        axs[3].set_title('ML prediction on running diff - residuals')
        axs[3].norm = matplotlib.colors.LogNorm()
        ims = []
        for fp in fits_points:
            filename = fp[0]
            points = fp[1]
            date = fp[2]
            hdul = fits.open(self.fits_directory + filename)
            fits_date = datetime.strptime(f"{hdul[0].header['DATE-OBS']} {hdul[0].header['TIME-OBS']}", '%Y/%m/%d %H:%M:%S.%f')
            fits_detector = hdul[0].header['DETECTOR']
            image = hdul[0].data.astype(np.float32)
            image = normalize_by_corner_statistics(image)

            diff = image.astype(np.float32)
            previous_fits = self.fits_repository.get_previous_fits(filename)
            if previous_fits is not None:
                hdul_prev = fits.open(self.fits_directory + previous_fits.filename)
                image_prev = hdul_prev[0].data.astype(np.float32)
                image_prev = normalize_by_corner_statistics(image_prev)

                prev_date = datetime.strptime(f"{hdul_prev[0].header['DATE-OBS']} {hdul_prev[0].header['TIME-OBS']}", '%Y/%m/%d %H:%M:%S.%f')
                prev_detector = hdul_prev[0].header['DETECTOR']
                diff = np.subtract(diff, image_prev)
                diff = normalize_by_corner_statistics(diff)
                print(f"PREV {fits_detector} {previous_fits.filename} ({prev_date.strftime('%Y-%m-%d %H:%M:%S.%f')})")
                print(f"CURR {prev_detector} {filename} ({fits_date.strftime('%Y-%m-%d %H:%M:%S.%f')}) time diff: {(fits_date - prev_date)} detector diff: {fits_detector}  {prev_detector}")
            else:
                print(f"No previous fits for {filename}")

            im1 = axs[0].imshow(image, animated=True, cmap="viridis", vmin=0, vmax=1, origin='lower')
            imdiff = axs[1].imshow(diff, animated=True, cmap="inferno", vmin=-1, vmax=1, origin='lower')

            im2 = None
            im2_residuals = None
            if model_path is not None:
                #to pytorch tensor
                model_input = torch.from_numpy(diff)
                model_output = model_inference.do_inference(model_input.float().unsqueeze(0).unsqueeze(0))
                mask = model_output.squeeze(0).squeeze(0).detach().numpy()

                im2 = axs[2].imshow(mask, animated=True, origin='lower')

                rows = [p[0] for p in points]
                cols = [p[1] for p in points]
                label = self.labeler.label(rows, cols, hdul[0].header)
                residuals = np.subtract(mask, label)
                im2_residuals = axs[3].imshow(residuals, animated=True, origin='lower')
            else:
                pass

            artists = [im1, imdiff]

            if im2 is not None:
                artists.append(im2)
                artists.append(im2_residuals)

            # add points to im plot
            for p in points:
                coord_x = p[1]
                coord_y = p[0]

                coord_x, coord_y = rotate_by_fits_header(coord_x, coord_y, hdul[0].header)

                point = axs[0].plot(coord_x, coord_y, 'o', color=p[2])
                artists.append(point[0])  # add point artist to the list
                point = axs[1].plot(coord_x, coord_y, 'o', color=p[2])
                artists.append(point[0])  # add point artist to the list

            # Store all artists for this frame
            ims.append(artists)

        fig.colorbar(im1, ax=axs[0], orientation='vertical')
        fig.colorbar(imdiff, ax=axs[1], orientation='vertical')
        if im2 is not None:
            fig.colorbar(im2, ax=axs[2], orientation='vertical')

        ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True)
        plt.show()
