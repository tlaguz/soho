import matplotlib
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

        # Plot the input.
        im1 = axs[0].imshow(input, cmap="inferno", vmin=-1, vmax=1, origin='lower')
        axs[0].set_title('Input - ' + filename)
        fig.colorbar(im1, ax=axs[0], orientation='vertical')
        print(filename)

        # Plot the label.
        im2 = axs[1].imshow(label, cmap='gray', vmin=0.0, vmax=1.00, origin='lower')
        axs[1].set_title('Label')
        fig.colorbar(im2, ax=axs[1], orientation='vertical')

        # Plot the output.
        im3 = axs[2].imshow(output, cmap='gray', vmin=0.0, vmax=1.00, origin='lower')
        axs[2].set_title('Output')
        fig.colorbar(im3, ax=axs[2], orientation='vertical')

        plt.show()

    def preview_metadata(self, metadata_file):
        training_metadata = TrainingMetadata(metadata_file)
        data_list = training_metadata.read()

        local_ranks = set([x.local_rank for x in data_list])

        fig, axs = plt.subplots(1, 3, figsize=(20, 10))  # Changed from 1,1 to 1,2 to create two columns

        # Plot for Epoch Loss
        axs[0].set_title('Epoch Loss')
        axs[0].set_xlabel('Iteration')
        axs[0].set_ylabel('Loss')
        for local_rank in local_ranks:
            axs[0].plot([x.epoch_loss for x in data_list if x.local_rank == local_rank])

        # Additional plot for x.loss
        axs[1].set_title('Loss for each iteration')
        axs[1].set_xlabel('Iteration')
        axs[1].set_ylabel('Loss')
        for local_rank in local_ranks:
            axs[1].plot([x.loss for x in data_list if
                         x.local_rank == local_rank])

        # Additional plot for x.loss
        axs[2].set_title('Validation loss')
        axs[2].set_xlabel('Iteration')
        axs[2].set_ylabel('Loss')
        axs[2].plot([x.valid_loss for x in data_list if
                     x.local_rank == 0])

        plt.show()
