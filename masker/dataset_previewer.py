import tkinter as tk
from tkinter import ttk

import numpy as np
import torch
from matplotlib import cm, animation
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
NavigationToolbar2Tk)

from masker.model_inference import ModelInference
from masker.normalizers.background_normalizer import BackgroundNormalizer
from masker.trainer_config import create_dataset


class DatasetPreviewer(tk.Tk):
    def __init__(self, dbconn, model):
        tk.Tk.__init__(self)

        self.model = model
        self.dbconn = dbconn
        self.dataset = create_dataset(dbconn, augment=False)

        self.title("Dataset preview")
        self.geometry("1600x800")

        self.label_top_title = tk.Text(self, height=1, borderwidth=0, highlightthickness=0, relief='flat', bg="white", font=("Arial", 16))
        self.label_top_title.tag_configure("text", foreground="black")
        self.label_top_title.pack(fill=tk.X, side=tk.TOP)

        # Create figure and axes here
        self.figure = Figure(figsize=(5, 5))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.draw()

        toolbar = NavigationToolbar2Tk(self.canvas, self)
        toolbar.update()

        self.button_refresh = tk.Button(self, text="Refresh", command=self.update_plot)
        self.button_next = tk.Button(self, text="Next", command=self.next_plot)
        self.button_previous = tk.Button(self, text="Previous", command=self.prev_plot)
        self.button_model = tk.Button(self, text="Run model", command=self.run_model)

        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.button_refresh.pack(side=tk.RIGHT)
        self.button_next.pack(side=tk.RIGHT)
        self.button_previous.pack(side=tk.RIGHT)
        self.button_model.pack(side=tk.RIGHT)

        self.combo_scale = ttk.Combobox(self, values=["linear", "log", "asinh", "sqrt"])
        self.combo_scale.current(0)
        self.combo_scale.pack(side=tk.RIGHT)
        self.combo_scale.bind("<<ComboboxSelected>>", lambda e: self.update_plot())

        self.dataset_id = 0
        self.dataset_item = None, None, None

        self.label_title = tk.Label(self, text="", font=("Arial", 16))
        self.label_title.pack(side=tk.TOP, fill=tk.X)

        self.update_plot()

    def _rescale(self, image):
        scale = self.combo_scale.get()
        if scale == "linear":
            return image
        elif scale == "log":
            return np.log(image + 1)
        elif scale == "asinh":
            return np.arcsinh(image)
        elif scale == "sqrt":
            return np.sqrt(image)
        else:
            print(f"Unknown scale {scale}")
            return image

    def next_plot(self):
        self.dataset_id = (self.dataset_id + 1) % len(self.dataset)
        self.update_plot()

    def prev_plot(self):
        self.dataset_id = (self.dataset_id - 1) % len(self.dataset)
        self.update_plot()

    def run_model(self):
        self.model_inference = ModelInference(self.model)

        input, label, txt = self.dataset_item
        if isinstance(input, list):
            input = [torch.from_numpy(i) for i in input]
            model_input = torch.stack(input)
        else:
            model_input = torch.from_numpy(input)

        model_output, loss, label = self.model_inference.do_inference(model_input, label)
        output = torch.sigmoid(model_output).detach().numpy()
        output = output.squeeze(0)
        imoutput = self.ax_model_output.imshow(output[0,:], vmin=0.0, vmax=1.0, origin='lower', animated=True)

        def update(i):
            imoutput.set_array(output[i,:])

        ani = animation.FuncAnimation(self.figure, update, frames=output.shape[0], repeat=True, interval=500)

        if self.ax_model_output_colorbar is not None:
            self.ax_model_output_colorbar.remove()
        if self.ax_model_output_text is not None:
            self.ax_model_output_text.remove()

        self.ax_model_output_colorbar = self.figure.colorbar(imoutput, ax=self.ax_model_output)
        self.ax_model_output_text = self.ax_model_output.text(0, 0, f"Loss {loss.item()}", va="bottom", ha="left", color="yellow")
        self.canvas.draw()

    def update_plot(self):
        self.figure.clear()  # Clear whole figure before creating new plots

        # Create new axes after clearing figure
        axes = [self.figure.add_subplot(131), self.figure.add_subplot(132), self.figure.add_subplot(133)]

        ax_running_diff = axes[0]
        ax_label = axes[1]
        ax_model_output = axes[2]

        self.dataset_item = self.dataset[self.dataset_id]
        input, label, txt = self.dataset_item

        ax_running_diff.set_title('Running diff')
        if isinstance(input, list):
            input = input[::-1]
            imdiff = ax_running_diff.imshow(input[0], cmap="inferno", vmin=-0.5, vmax=0.5, origin='lower')
            self.figure.colorbar(imdiff, ax=ax_running_diff)

            text = ax_running_diff.text(0.5, 0.5, "Frame 0", verticalalignment='center', horizontalalignment='center',
                           transform=ax_running_diff.transAxes, color='white')

            def update(i):
                data = self._rescale(input[i])
                imdiff.set_array(data)
                if i == len(input) - 1:
                    text.set_text("Frame " + str(i+1) + " (LAST)")
                else:
                    text.set_text("Frame " + str(i+1))

            ani = animation.FuncAnimation(self.figure, update, frames=len(input), repeat=True, interval=500)
        else:
            imdiff = ax_running_diff.imshow(self._rescale(input), cmap="inferno", vmin=-0.5, vmax=0.5, origin='lower')
            self.figure.colorbar(imdiff, ax=ax_running_diff)

        imlabel = ax_label.imshow(label, cmap='gray', vmin=0.0, vmax=1.00, origin='lower')
        self.figure.colorbar(imlabel, ax=ax_label)
        ax_label.set_title('Label')

        self.ax_model_output = ax_model_output
        ax_model_output.set_title('Model output')

        self.ax_model_output_colorbar = None
        self.ax_model_output_text = None

        self.canvas.draw()
        self.label_top_title.insert(1.0, txt, "text")
        self.label_title.config(text=f"Currently displaying: {self.dataset_id + 1}")

