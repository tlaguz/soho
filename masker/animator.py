from datetime import datetime

import numpy as np

import PyQt5

import matplotlib
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from masker.fits_loader import FitsLoader
from masker.model_inference import ModelInference
from masker.repositories.fits_repository import FitsRepository
from masker.trainer_config import create_labeler, create_normalizer
from masker.utils import rotate_by_fits_header

matplotlib.use("TkAgg")
#matplotlib.use('Qt5Agg')

# Set up matplotlib
import matplotlib.pyplot as plt

from matplotlib import animation


class Animator:
    def __init__(self, dbconn):
        self.dbconn = dbconn
        self.fits_loader = FitsLoader()
        self.fits_repository = FitsRepository(dbconn)
        self.labeler = create_labeler()
        self.normalizer = create_normalizer()

    def _show_ex_pdf(self, mask, axs, axs_statistics_x, axs_statistics_y):
        # Normalize probabilities
        probabilities = mask / mask.sum()

        # Marginal distributions
        x_distribution = np.sum(probabilities, axis=0)
        y_distribution = np.sum(probabilities, axis=1)

        flat_index = np.argmax(probabilities)
        max_y, max_x = np.unravel_index(flat_index, probabilities.shape)

        # Expected values
        expected_x = np.sum(np.arange(probabilities.shape[1]) * x_distribution)
        expected_y = np.sum(np.arange(probabilities.shape[0]) * y_distribution)

        # Main 2D array
        img = axs.imshow(mask, origin='lower', cmap='viridis', animated=True, norm=matplotlib.colors.PowerNorm(vmin=0, vmax=1, gamma=2))
        img_exh = axs.axhline(expected_y, color='r', animated=True)
        img_exv = axs.axvline(expected_x, color='r', animated=True)
        img_maxh = axs.axhline(max_y, color='y', animated=True)
        img_maxv = axs.axvline(max_x, color='y', animated=True)

        # X marginal distribution
        img_x_bar = axs_statistics_x.bar(range(len(x_distribution)), x_distribution, color='g', animated=True)
        img_x = axs_statistics_x.scatter(expected_x, x_distribution[int(expected_x)], color='r', zorder=5, animated=True)

        # Y marginal distribution
        img_y_bar = axs_statistics_y.barh(range(len(y_distribution)), y_distribution, color='b', animated=True)
        img_y = axs_statistics_y.scatter(y_distribution[int(expected_y)], expected_y, color='r', zorder=5, animated=True)

        # Hide labels for marginal distribution plots
        plt.setp(axs_statistics_x.get_xticklabels(), visible=False)
        plt.setp(axs_statistics_y.get_yticklabels(), visible=False)

        result = [img, img_exh, img_exv, img_x, img_y, img_maxh, img_maxv]
        result.extend(img_x_bar)
        result.extend(img_y_bar)
        return img, result

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

        fig, axs = plt.subplots(2, 3, figsize=(20, 10))

        fig.suptitle(' ; '.join(yhts))

        axs_raw_manual_label = axs[0, 0]
        axs_running_diff_manual_label = axs[1, 0]

        axs_training_label = axs[0, 1]
        axs_statistics = axs[1, 1]
        # Define axes for marginal distributions around axs[1, 1]
        divider = make_axes_locatable(axs_statistics)
        axs_statistics_x = divider.append_axes("top", 0.6, pad=0.1, sharex=axs_statistics)
        axs_statistics_y = divider.append_axes("left", 0.6, pad=0.1, sharey=axs_statistics)
        # Rotate the y distribution axes for vertical
        axs_statistics_y.invert_xaxis()

        axs_ml_prediction = axs[0, 2]
        axs_ml_prediction_residuals = axs[1, 2]

        axs_raw_manual_label.set_title('Raw with manual label')
        axs_running_diff_manual_label.set_title('Running diff with manual label')
        axs_ml_prediction.set_title('ML prediction on running diff')
        axs_ml_prediction_residuals.set_title('ML prediction on running diff - residuals')
        axs_training_label.set_title('Training label')

        ims = []
        for fp in fits_points:
            filename = fp[0]
            points = fp[1]
            date = fp[2]
            hdul = self.fits_loader.get_cached_fits(filename)
            fits_date = datetime.strptime(f"{hdul[0].header['DATE-OBS']} {hdul[0].header['TIME-OBS']}", '%Y/%m/%d %H:%M:%S.%f')
            fits_detector = hdul[0].header['DETECTOR']
            exptime = hdul[0].header['EXPTIME']
            exp0 = hdul[0].header['EXP0']
            image = hdul[0].data.astype(np.float32)
            image = self.normalizer.normalize_observation(image, hdul[0].header)

            diff = image
            previous_fits = self.fits_repository.get_previous_fits(filename)
            if previous_fits is not None:
                hdul_prev = self.fits_loader.get_cached_fits(previous_fits.filename)
                image_prev = hdul_prev[0].data.astype(np.float32)
                image_prev = self.normalizer.normalize_observation(image_prev, hdul_prev[0].header)

                prev_date = datetime.strptime(f"{hdul_prev[0].header['DATE-OBS']} {hdul_prev[0].header['TIME-OBS']}", '%Y/%m/%d %H:%M:%S.%f')
                prev_detector = hdul_prev[0].header['DETECTOR']
                diff = np.subtract(diff, image_prev)
                diff = self.normalizer.normalize_diff(diff, hdul[0].header)
                print(f"PREV {fits_detector} {previous_fits.filename} ({prev_date.strftime('%Y-%m-%d %H:%M:%S.%f')})")
                print(f"CURR {prev_detector} {filename} ({fits_date.strftime('%Y-%m-%d %H:%M:%S.%f')}) time diff: {(fits_date - prev_date)} detector diff: {fits_detector}  {prev_detector} ; exptime: {exptime} exp0: {exp0}")
            else:
                print(f"No previous fits for {filename}")

            im1 = axs_raw_manual_label.imshow(image, animated=True, cmap="viridis", vmin=-20, vmax=20, origin='lower')
            imdiff = axs_running_diff_manual_label.imshow(diff, animated=True, cmap="inferno", vmin=-0.5, vmax=0.5, origin='lower')

            im2 = None
            im2_text = None
            im2_residuals = None
            im_label = None
            img_statistics = None
            img_statistics_list = []
            if model_path is not None:
                rows = [p[0] for p in points]
                cols = [p[1] for p in points]
                label = self.labeler.label(rows, cols, hdul[0].header)
                label = self.normalizer.normalize_label(label)

                #to pytorch tensor
                model_input = torch.from_numpy(diff)
                model_output, loss, label = model_inference.do_inference(model_input.unsqueeze(0).unsqueeze(0), label)
                model_output = torch.sigmoid(model_output)
                mask = model_output.squeeze(0).squeeze(0).detach().numpy()

                im2 = axs_ml_prediction.imshow(mask, animated=True, origin='lower')
                im2_text = axs_ml_prediction.text(0, 0, f"Loss {loss.item()}", va="bottom", ha="left", color="yellow")

                residuals = np.subtract(mask, label.squeeze(0))
                im2_residuals = axs_ml_prediction_residuals.imshow(residuals[0], vmin=-1, vmax=1, animated=True, origin='lower')

                im_label = axs_training_label.imshow(label.squeeze(0)[0], vmin=0, vmax=1, animated=True, origin='lower')

                img_statistics, img_statistics_list = self._show_ex_pdf(mask, axs_statistics, axs_statistics_x, axs_statistics_y)
            else:
                pass

            artists = [im1, imdiff]

            if im2 is not None:
                artists.append(im2)
                artists.append(im2_text)
                artists.append(im2_residuals)
                artists.append(im_label)
                artists.extend(img_statistics_list)

            # add points to im plot
            for p in points:
                coord_x = p[1]
                coord_y = p[0]

                coord_x, coord_y = rotate_by_fits_header(coord_x, coord_y, hdul[0].header)

                point = axs_raw_manual_label.plot(coord_x, coord_y, 'o', color=p[2])
                artists.append(point[0])  # add point artist to the list
                point = axs_running_diff_manual_label.plot(coord_x, coord_y, 'o', color=p[2])
                artists.append(point[0])  # add point artist to the list

            # Store all artists for this frame
            ims.append(artists)

        fig.colorbar(im1, ax=axs_raw_manual_label, orientation='vertical')
        fig.colorbar(imdiff, ax=axs_running_diff_manual_label, orientation='vertical')
        if im2 is not None:
            fig.colorbar(im2, ax=axs_ml_prediction, orientation='vertical')
            fig.colorbar(img_statistics, ax=axs_statistics, orientation='vertical')
        if im2_residuals is not None:
            fig.colorbar(im2_residuals, ax=axs_ml_prediction_residuals, orientation='vertical')
        if im_label is not None:
            fig.colorbar(im_label, ax=axs_training_label, orientation='vertical')

        ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True)

        ani.save("animation.gif", writer='imagemagick')

        plt.show()
