import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt

from masker.training_metadata import TrainingMetadata


class Previewer:
    def __init__(self):
        pass

    # input, label and output are all numpy arrays or torch tensors.
    # Each are 1024x1024, float32.
    def preview(self, input, label, filename, output):
        matplotlib.use("TkAgg")

        if isinstance(input, torch.Tensor):
            input = input.cpu().detach().numpy()
        if isinstance(label, torch.Tensor):
            label = label.cpu().detach().numpy()
        if isinstance(output, torch.Tensor):
            output = output.cpu().detach().numpy()

        # Create a 3x1 grid of plots.
        fig, axs = plt.subplots(3, 1, figsize=(10, 10))

        axs_input = axs[0]
        axs_label = axs[1]
        axs_output = axs[2]

        # Plot the input.
        im1 = axs_input.imshow(input, cmap="inferno", vmin=-1, vmax=1, origin='lower')
        axs_input.set_title('Input - ' + filename)
        fig.colorbar(im1, ax=axs_input, orientation='vertical')
        print(filename)

        # Plot the label.
        im2 = axs_label.imshow(label, cmap='gray', vmin=0.0, vmax=1.00, origin='lower')
        axs_label.set_title('Label')
        fig.colorbar(im2, ax=axs_label, orientation='vertical')

        # Plot the output.
        im3 = axs_output.imshow(output, cmap='gray', vmin=0.0, vmax=1.00, origin='lower')
        axs_output.set_title('Output')
        fig.colorbar(im3, ax=axs_output, orientation='vertical')

        plt.show()

    def preview_metadata(self, metadata_file):
        training_metadata = TrainingMetadata(metadata_file)
        data_list = training_metadata.read()

        local_ranks = set([x.local_rank for x in data_list])

        fig, axs = plt.subplots(1, 4, figsize=(20, 10))

        axs_loss = axs[0]
        axs_iter_loss = axs[1]
        axs_valid_loss = axs[2]
        axs_lr = axs[3]

        previous_epochs = {}  # Previous epoch for each local rank
        losses_per_epoch = {}  # Average of losses for each epoch

        for i, obj in enumerate(data_list):
            if obj.local_rank not in previous_epochs or obj.epoch > previous_epochs[obj.local_rank]:
                previous_epochs[obj.local_rank] = obj.epoch

                if losses_per_epoch.get(obj.epoch) is None:
                    losses_per_epoch[obj.epoch] = []

                losses_per_epoch[obj.epoch].append((i, obj.epoch_loss))

        epoch_x_coords = []
        epoch_y_coords = []
        for epoch, loss_data in losses_per_epoch.items():
            i, epoch_loss = zip(*loss_data)
            epoch_x_coords.append(np.average(i)/len(local_ranks))
            epoch_y_coords.append(sum(epoch_loss) / len(epoch_loss))

        axs_loss.scatter(epoch_x_coords, epoch_y_coords, color='red')

        axs_loss.set_title('Epoch Loss')
        axs_loss.set_xlabel('Iteration')
        axs_loss.set_ylabel('Loss')
        for local_rank in local_ranks:
            axs_loss.plot([x.epoch_loss for x in data_list if x.local_rank == local_rank])

        axs_iter_loss.set_title('Loss for each iteration')
        axs_iter_loss.set_xlabel('Iteration')
        axs_iter_loss.set_ylabel('Loss')
        for local_rank in local_ranks:
            axs_iter_loss.plot([x.loss for x in data_list if
                         x.local_rank == local_rank])

        axs_valid_loss.set_title('Validation loss')
        axs_valid_loss.set_xlabel('Iteration')
        axs_valid_loss.set_ylabel('Loss')
        axs_valid_loss.plot([x.valid_loss for x in data_list if
                     x.local_rank == 0])

        axs_lr.set_title('Learning rate')
        axs_lr.set_xlabel('Iteration')
        axs_lr.set_ylabel('Learning rate')
        axs_lr.plot([x.lr for x in data_list if
             x.local_rank == 0])

        plt.show()
